from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

# 1. Core Input Model (The client request for classification) 

class TransactionIn(BaseModel):
    """
    Schema for an incoming transaction request to the /classify endpoint.
    This is what the client sends.
    """
    raw_description: str = Field(..., min_length=3, description="The raw, unclassified merchant description.")
    amount: float = Field(..., gt=0, description="The transaction amount (must be positive).")
    date: Optional[datetime] = Field(None, description="Optional date/time of the transaction.")

# 2. Core Output Model (The API response after classification) 

class ClassificationOut(BaseModel):
    """
    Schema for the response from the /classify endpoint.
    This includes the classification results.
    """
    transaction_id: Optional[str] = Field(None, description="Internal system ID, if logged.")
    raw_description: str
    predicted_category: str = Field(..., description="The predicted financial category.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model's confidence (0.0 to 1.0).")
    classification_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Time of classification.")

# 3. Database Model (If you use a MongoDB/SQL-like storage) 

class TransactionDB(ClassificationOut):
    """
    Schema for storing the transaction record in the database.
    Inherits from ClassificationOut but adds essential DB fields.
    """
    # Assuming primary key is a string (e.g., UUID or MongoDB ObjectID)
    transaction_id: str = Field(..., description="Unique ID for the database record.")
    user_id: Optional[str] = Field(None, description="ID of the user this transaction belongs to.")
    
    # Fields that might be needed for the retraining loop
    is_reviewed: bool = Field(False, description="Flag if classification was manually checked/corrected.")
    correct_category: Optional[str] = Field(None, description="The manually corrected category (if reviewed).")

    class Config:
        # Allows Pydantic to work with ORMs/databases that use attribute access (SQLAlchemy)
        from_attributes = True 
        # Optional: Example for a MongoDB ID field handling
        # json_encoders = {ObjectId: str}
