from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import silhouette_score

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

from typing import List
import random
import pickle

from encoder import SiameseEncoder

class ImageRecommendation:
    """
    Holds the sources images and their recommendation images from the base dataset.
    Contains the functionality for displaying the recommendations to the user.
    """
    def __init__(self, src_images: List[str], rec_images: List[List[str]]) -> None:
        self.src_images = src_images
        self.rec_images = rec_images

    def show_recommendations_group(self, index: int, max_recommend: int) -> None:
        if max_recommend <= 1:
            print("Using recommendation group requires 'max_recommend' of at least 2.")
            return

        self._show_image(self.src_images[index])

        if max_recommend < len(self.rec_images[index]):
            images = random.sample(self.rec_images[index], max_recommend)
            self._show_image_group(images)
            return

        self._show_image_group(self.rec_images[index])

    def show_recommendations_individual(self, index: int, max_recommend: int) -> None:
        if max_recommend < 1:
            print("Using recommendation requires 'max_recommend' of at least 1.")
            return

        self._show_image(self.src_images[index])

        if max_recommend < len(self.rec_images[index]):
            images = random.sample(self.rec_images[index], max_recommend)
            for image in images:
                self._show_image(image)
            return
        
        for image in self.rec_images[index]:
            self._show_image(image)

    def _show_image(self, image_path: str) -> None:
        image = Image.open(image_path)

        plt.imshow(image) # type: ignore
        plt.axis("off")
        plt.show()

    def _show_image_group(self, image_paths: List[str]) -> None:
        num_images = len(image_paths)

        _, axes = plt.subplots(1, num_images)

        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path)

            axes[i].imshow(image)
            axes[i].axis("off")
        
        plt.show()

class ImagePreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformer for input preparation.

    Input: List of image paths.
    Output: List of tensors.
    """
    def __init__(self) -> None:
        super(ImagePreprocessor, self).__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, images: List[str]):
        print("Transforming input")

        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()
        ])

        tensor_images = []
        for path in images:
            image = Image.open(path)
            image = image.convert("RGB")
            tensor_images.append(transform(image))
            
        return tensor_images
    
class Encoder(BaseEstimator, TransformerMixin):
    """
    Transformer for encoding the inputs.

    Input: List of tensors.
    Output: 2D numpy array of the encoded inputs.
    """
    def __init__(self, encoder: str, save_tensors: bool = False) -> None:
        super(Encoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_tensors = save_tensors

        self.encoder = SiameseEncoder().to(self.device)
        if torch.cuda.is_available():
            self.encoder.load_state_dict(torch.load(encoder))
        else:
            self.encoder.load_state_dict(torch.load(encoder, map_location=torch.device("cpu")))

        self.encoder.eval()
        
    def fit(self, images: List[torch.Tensor], y=None):
        if not self.save_tensors:
            return self
        
        print("Encoding: Fitting and saving database tensors")
        output_tensor_list = []

        with torch.no_grad():
            for image in images:
                input = image.to(self.device)
                
                # (3, 300, 300) -> (1, 32)
                output = self.encoder.forward_once(input.unsqueeze(0))
                output_tensor_list.append(output)

        # (len, 1, 32)
        self.database_tensors = torch.stack(output_tensor_list).cpu()

        return self
    
    def transform(self, images: List[torch.Tensor]):
        print("Encoding")
        output_tensor_list = []

        with torch.no_grad():
            for image in images:
                input = image.to(self.device)
                
                # (3, 300, 300) -> (1, 32)
                output = self.encoder.forward_once(input.unsqueeze(0))
                output_tensor_list.append(output)

        # List[ (1, 32) ] -> numpy(len, 32)
        encoded_array = torch.stack(output_tensor_list).squeeze(dim=1).cpu().numpy()
        
        print("Clustering")
        return encoded_array
    
class ArtisticImageRecommender:
    """
    Image recommender trained on a dataset of surreal diverse art.
    """
    def __init__(self, encoder: str, dataset: str, classes: int, save_tensors: bool = False) -> None:
        self.encoder_path = encoder

        image_folder = ImageFolder(dataset)
        self.images = [path for path, _ in image_folder.imgs]

        self.num_classes = classes
        self.save_tensors = save_tensors

        steps = [
            ("preprocess", ImagePreprocessor()),
            ("encode",     Encoder(self.encoder_path, self.save_tensors)),
            ("cluster",    KMeans(n_clusters=classes, n_init="auto"))
        ]
        self.pipe = Pipeline(steps)

    def fit(self) -> None:
        output = self.pipe.fit(self.images)
        self.cluster_labels = output.named_steps["cluster"].labels_

    def recommend(self, images_path: str) -> ImageRecommendation:
        image_folder = ImageFolder(images_path)
        images = [path for path, _ in image_folder.imgs]

        output = self.pipe.predict(images)

        recommendations = []
        for cluster in output:
            images_from_cluster = [self.images[i] for i, label in enumerate(self.cluster_labels) if label == cluster]
            recommendations.append(images_from_cluster)

        return ImageRecommendation(images, recommendations)
    
    def advanced_recommend(self, image_path: str, num_recommend: int):
        """
        Only works on one image. Also 'save_database_tensors' must be set to true.
        """
        if not self.save_tensors:
            print("'save_database_tensors' must be equal to True.")
            return

        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()
        ])

        image = Image.open(image_path)
        image = image.convert("RGB")
        image = transform(image)

        device = self.pipe.named_steps["encode"].device
        with torch.no_grad():
            image = image.to(device) # type: ignore
            image_embedding = self.pipe.named_steps["encode"].encoder.forward_once(image.unsqueeze(0))

        recommendations = self._top_k_matches(image_embedding, num_recommend)
        
        return ImageRecommendation([image_path], [recommendations])
    
    def optimize_clustering(self, ) -> None:
        """
        Used for optimizing the clustering process. Does not interact with the model but
        instead just creates its own KMeans instance to test on. This is so you can avoid
        having to do all of the data preparation and encoding every time. But, that does
        mean 'save_database_tensors' does need to be set to True because the KMeans will
        use this as its testing data.
        """
        if not self.save_tensors:
            print("'save_database_tensors' must be equal to True.")
            return
        
        clustering_data = self.pipe.named_steps["encode"].database_tensors
        clustering_data = clustering_data.squeeze(dim=1).numpy()

        print(f"Instances to cluster: {len(clustering_data)}")

        for k in range(15, 75):
            k_means = KMeans(n_clusters=k, n_init="auto")

            cluster_labels = k_means.fit_predict(clustering_data)

            score = silhouette_score(clustering_data, cluster_labels)
            print(f"k: {k}, score: {score}")

        return

    def _top_k_matches(self, input: torch.Tensor, k: int) -> List[str]:
        device = self.pipe.named_steps["encode"].device
        database_tensors = self.pipe.named_steps["encode"].database_tensors.squeeze(dim=1).to(device)
        
        # Broadcast input so that it is the same size as the database_tensor
        input_broadcast = input.expand(database_tensors.shape[0], -1)

        distances = F.pairwise_distance(input_broadcast, database_tensors)

        _, indices = torch.topk(distances, k=k, largest=False)
        indices = indices.cpu().tolist()

        return [self.images[i] for i in indices]

def build_model(
        encoder:               str, 
        dataset:               str, 
        num_classes:           int, 
        save_model:            bool, 
        save_model_path:       str,
        save_database_tensors: bool = False
    ) -> ArtisticImageRecommender:
    
    model = ArtisticImageRecommender(encoder, dataset, num_classes, save_database_tensors)
    model.fit()

    if save_model:
        with open(save_model_path, "wb") as f:
            pickle.dump(model, f)

    return model

def load_model(model: str) -> ArtisticImageRecommender:
    with open(model, "rb") as f:
        saved_model = pickle.load(f)

    return saved_model

def save_model_cpu(model: ArtisticImageRecommender, save_model_path: str):
    model.pipe.named_steps["encode"].device = "cpu"
    model.pipe.named_steps["encode"].encoder.cpu()

    with open(save_model_path, "wb") as f:
        pickle.dump(model, f)