from datetime import datetime
from pydantic import BaseModel, Field, BeforeValidator
from typing import List, Optional, Annotated

# Custom type for ObjectId to str conversion
PyObjectId = Annotated[str, BeforeValidator(str)]


# Nested models for populated fields
class UserResponse(BaseModel):
    id: PyObjectId = Field(..., alias="_id")
    username: Optional[str] = None
    image: Optional[str] = None
    phone: Optional[str] = None  # Added phone based on product's userId populate

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class CategoryResponse(BaseModel):
    id: PyObjectId = Field(..., alias="_id")
    name: Optional[str] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class ProductResponse(BaseModel):
    id: PyObjectId = Field(..., alias="_id")
    name: Optional[str] = None
    image: Optional[str] = None
    price: Optional[float] = None  # Assuming price is a float
    category: Optional[CategoryResponse] = None
    userId: Optional[UserResponse] = None  # This is the user who owns the product

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class VideoResponse(BaseModel):
    id: PyObjectId = Field(..., alias="_id")
    userId: UserResponse
    targetUserId: Optional[UserResponse] = None
    url: str
    filename: str  # Added filename
    direction: Optional[str] = None  # Added direction
    title: Optional[str] = None
    description: Optional[str] = None
    thumbnail: Optional[str] = None
    hashtags: List[str] = []
    products: List[ProductResponse] = []
    visibility: str  # Added visibility
    views: int = 0
    manifestUrl: Optional[str] = None  # Added manifestUrl
    status: str  # Added status
    duration: float = 0.0
    createdAt: datetime = Field(..., alias="createdAt")
    updatedAt: datetime = Field(..., alias="updatedAt")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_schema_extra = {}
