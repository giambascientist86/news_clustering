from pydantic import BaseModel, Field
from typing import List

class TextInput(BaseModel):
    texts: List[str] = Field(..., description="Lista di stringhe di input per il modello.")

class TopicModel(BaseModel):
    Topic: int = Field(..., alias="Topic", description="Identificatore numerico del topic.")
    Count: int = Field(..., alias="Count", description="Numero di documenti appartenenti al topic.")
    Name: str = Field(..., alias="Name", description="Nome del topic.")
    Representation: List[str] = Field(..., alias="Representation", default_factory=list, description="Lista di parole rappresentative del topic.")
    KeyBERTInspired: List[str] = Field(..., alias="KeyBERTInspired", default_factory=list, description="Lista di parole chiave estratte con KeyBERT.")
    MMR: List[str] = Field(..., alias="MMR", default_factory=list, description="Lista di parole chiave selezionate con il metodo MMR.")
    Representative_Docs: List[str] = Field(..., alias="Representative_Docs", default_factory=list, description="Lista di documenti rappresentativi del topic.")

    class Config:
        populate_by_name = True  # Permette di usare gli alias nei JSON di risposta
        arbitrary_types_allowed = True  # Evita errori se servissero tipi non nativi


