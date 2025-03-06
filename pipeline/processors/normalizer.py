# Data normalization processor
from .base import DataProcessor
from typing import Dict, Any, List, Tuple
from ..processors.base import DataProcessor
from ..core.data_model import TextItem, CategoryResult
import pandas as pd
import re
from tqdm import tqdm
import logging
import Levenshtein

logger = logging.getLogger(__name__)

class Normalizer(DataProcessor):
    """
    A processor that normalizes data according to specified rules.
    """
    
    def __init__(self, data_path: str, edam_topics_path: str):
        self.data_path = data_path
        self.edam_topics_path = edam_topics_path
        self.edam_topics = self.load_edam_topics()
        self.quoted_topics = self.load_quoted_topics()
        self.synonym_dict = self.load_synonyms()

    def load_edam_topics(self) -> List[str]:
        """Load EDAM topics from a file."""
        with open(self.edam_topics_path, 'r') as f:
            edam_topics = [topic.strip() for topic in f.readlines()]
        return [topic[1:-1] if topic.startswith('"') and topic.endswith('"') else topic for topic in edam_topics]
    
    def load_quoted_topics(self) -> List[str]:
        return [topic for topic in self.edam_topics if topic.startswith('"') and topic.endswith('"')]

    def load_synonyms(self) -> dict:
        """Load synonyms from the EDAM dataset."""
        edam = pd.read_csv('EDAM/EDAM.csv')
        edam = edam[edam['Class ID'].str.contains('topic')].reset_index(drop=True)
        edam['Synonyms'] = edam['Synonyms'].fillna('').apply(lambda x: x.split('|') if x != '' else [])
        
        synonym_dict = {}
        for _, row in edam.iterrows():
            for synonym in row['Synonyms']:
                synonym_dict[synonym] = row['Preferred Label']
        return synonym_dict

    def split_topics(self, topics: str) -> List[str]:
        """Split and clean topics."""
        cleaned_topics = [topic.strip() for topic in topics.split('\t')]
        for i in range(len(cleaned_topics)):
            for quoted_topic in self.quoted_topics:
                if quoted_topic.replace('\"', '').lower() in cleaned_topics[i].lower():
                    cleaned_topics[i] = cleaned_topics[i].replace(quoted_topic.replace('\"', ''), quoted_topic)
                    break
                else:
                    cleaned_topics[i] = cleaned_topics[i].replace('\"', '')
        return cleaned_topics

    def process_predictions(self, predictions: List[str]) -> List[str]:
        """Process predictions to match EDAM topics."""
        processed_predictions = []
        for prediction in predictions:
            formatted = False
            for topic in self.quoted_topics:
                formatted_topic = topic.replace('\"', '')
                if formatted_topic in prediction:
                    processed_prediction = prediction.replace(formatted_topic, f'{topic}')
                    processed_predictions.append(processed_prediction)
                    formatted = True
                    break
            if not formatted:
                processed_predictions.append(prediction)
            
        final_predictions = []
        for prediction in processed_predictions:
            if '\"' in prediction:
                parts = re.findall(r'[^"]+|"[^"]+"', prediction)
                final_predictions.extend(parts)
            else:
                final_predictions.extend([pred.strip() for pred in prediction.split(',')])
        return final_predictions

    def clean_and_split(self, predictions: List[str]) -> List[str]:
        """Clean and split predictions based on various separators."""
        separators = ['    ', '   ', '  ', '\n', '<TAB>', 'TAB', '<tab>', '(tab)', '<Tab>', '[tab]', '▪️', '(Tab)', 
                      '\xa0\xa0\xa0\xa0', '\xa0', '\u2003', '、', '\x0b', '\x0c', ';', '.', '--', '-', '–', '_', 
                      '\\', '\\n', '/', '@', '|', '\r', '+', '<', '>', '·']

        # Join the separators with the regex OR operator
        sep_pattern = '|'.join(map(re.escape, separators))

        cleaned_predictions = []
        for pred in predictions:
            # Replace commas not enclosed in double quotes with |
            pred = re.sub(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', '|', pred)
            # Split on the separators
            split_list = re.split(sep_pattern, pred)
            # Flatten the list and strip whitespace
            cleaned_predictions.extend([item.strip() for item in split_list if item.strip()])

        # Remove 'Category X:' prefix if present
        cleaned_predictions = [re.sub(r'Category \d+:', '', pred) for pred in cleaned_predictions]
        return cleaned_predictions

    def identify_unexpected_predictions(self, results: List[CategoryResult]) -> List[Tuple[int, str]]:
        """Identify unexpected predictions that are not in the EDAM topics."""
        unexpected_predictions = []
        for index, result in enumerate(results):
            if result.model_response and result.categories:
                predictions = result.categories
                if len(predictions) <= 1:
                    prediction = predictions[0] if predictions else ""
                    if '\"' not in prediction and ' ' in prediction and prediction not in self.edam_topics:
                        unexpected_predictions.append((index, prediction))

        # Log unexpected predictions
        count = len(unexpected_predictions)
        logger.info(f"Number of unexpected predictions: {count}")
        for original_index, prediction in unexpected_predictions:
            logger.info(f"Original Index: {original_index}, Prediction: {prediction}")

        return unexpected_predictions
    
    

    def match_hallucinations(self, results: List[CategoryResult]) -> None:
        """Match hallucinations against EDAM topics and synonyms."""
        matched_topics = {}
        for result in results:
            if result.model_response and 'hallucinations' in result.model_response:
                hallucinations = result.model_response['hallucinations']
                for hallucination in hallucinations:
                    if hallucination in matched_topics:
                        continue
                    matched = False
                    # First check for a match in the topics list
                    sorted_topics = sorted(self.edam_topics, key=lambda topic: Levenshtein.distance(hallucination, topic))
                    for topic in sorted_topics:
                        distance = Levenshtein.distance(hallucination, topic)
                        if 0 < distance <= 2:
                            matched_topics[hallucination] = topic
                            matched = True
                            break
                    
                    # If no match in the topics list, look through the available synonyms
                    if not matched:
                        sorted_synonyms = sorted(self.synonym_dict.keys(), key=lambda topic: Levenshtein.distance(hallucination, topic))
                        for topic in sorted_synonyms:
                            distance = Levenshtein.distance(hallucination, topic)
                            if 0 <= distance <= 1:
                                matched_topics[hallucination] = self.synonym_dict[topic]
                                matched = True
                                break     

                    if matched:
                        continue

                    for topic in sorted_topics:
                        if topic.lower() in hallucination.lower().split():
                            matched_topics[hallucination] = topic
                            break
                    else:
                        for topic in sorted_synonyms:
                            if topic.lower() in hallucination.lower().split():
                                matched_topics[hallucination] = self.synonym_dict[topic]
                                break

        # Update predictions with matched topics
        for result in results:
            if result.model_response and 'hallucinations' in result.model_response:
                hallucinations = result.model_response['hallucinations']
                for hallucination in hallucinations:
                    if hallucination in matched_topics:
                        logger.info(f"'{hallucination}' matches topic '{matched_topics[hallucination]}'")
                        if result.categories:
                            result.categories.append(matched_topics[hallucination])

    def parse_hallucinations(self, results: List[CategoryResult]) -> None:
        """Parse hallucinations from categories and store them in model_response."""
        for result in results:
            if result.categories:
                hallucinations = set([cat.replace('.', '').replace('\"', '') for cat in result.categories if cat.replace('.', '').replace('\"', '') not in self.edam_topics])
                result.model_response['hallucinations'] = list(hallucinations)

                # Remove hallucinations from categories
                result.categories = [cat for cat in result.categories if cat.replace('.', '').replace('\"', '') not in hallucinations]                

    def normalize(self, results: List[CategoryResult]) -> None:
        """Main normalization process for CategoryResult objects."""
        for result in tqdm(results):
            if result.model_response and 'raw_response' in result.model_response:
                predictions = result.model_response['raw_response']
                # Normalize predictions
                predictions = self.split_topics(predictions)

                predictions = self.clean_and_split(predictions)
                predictions = self.process_predictions(predictions)
                result.categories = predictions

        
        # Identify unexpected predictions after normalization
        self.identify_unexpected_predictions(results)
        # Parse hallucinations from categories
        self.parse_hallucinations(results)
        # Match hallucinations against EDAM topics and synonyms
        self.match_hallucinations(results)

    def process_input(self, items: List[TextItem]) -> List[TextItem]:
        """Process input data before categorization."""
        return items

    def process_output(self, results: List[CategoryResult]) -> List[CategoryResult]:
        """Process results after categorization."""
        self.normalize(results)  # Normalize the results
        return results
    