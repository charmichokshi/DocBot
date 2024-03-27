import yaml
from pyprojroot import here


class LoadConfig:
    """
    A class for loading configuration settings, including Cohere API key.

    This class reads configuration parameters from a YAML file and sets them as attributes.
    It also includes a method to load Cohere API credentials.

    Attributes:
        cohere_embedding_model (str): The GPT model to be used.
        chunk_size (int): The chunk_size parameter for generating text tokens.
        chunk_overlap (int): The chunk_overlap parameter for generating text tokens.
        llm_system_role (str): The system role for the language model.

    Methods:
        __init__(): Initializes the LoadConfig instance by loading configuration from a YAML file.
        load_cohere_credentials(): Loads Cohere configuration settings.
    """

    def __init__(self) -> None:
        with open(here("config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.cohere_embedding_model = app_config["cohere_embedding_model"]
        self.chunk_size = app_config["chunk_size"]
        self.chunk_overlap = app_config["chunk_overlap"]
        self.llm_system_role = app_config["llm_system_role"]
        self.llm_format_output = app_config["llm_format_output"]
        # self.storage = app_config["storage"]
        # self.cache = app_config["cache"]
        self.llm_system_role_history = app_config['llm_system_role_history']
