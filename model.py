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


# Nested models for Product's images, sizes, and toppings
class ProductImageResponse(BaseModel):
    url: str
    filename: str

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class ProductSizeResponse(BaseModel):
    name: str
    extraCost: float = 0.0  # Mongoose schema defines as Number, default 0

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class ProductToppingResponse(BaseModel):
    name: str
    extraCost: float = 0.0  # Mongoose schema defines as Number, default 0, min 0
    isAvailable: bool = True  # Mongoose schema defines as Boolean, default true

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class ProductLocationResponse(BaseModel):
    type: str = "Point"  # Mongoose schema default "Point"
    coordinates: List[float]  # [longitude, latitude]

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class ProductResponse(BaseModel):
    id: PyObjectId = Field(..., alias="_id")
    userId: UserResponse  # Populated User object
    name: str
    cost: float  # Mongoose schema defines as Number
    currency: Optional[str] = "VND"  # Mongoose schema enum and default
    description: Optional[str] = None
    quantity: int = Field(
        ..., alias="qty"
    )  # Mongoose schema has alias "qty" for quantity
    category: Optional[CategoryResponse] = None  # Populated Category object
    ship: bool = True  # Mongoose schema default true
    images: List[ProductImageResponse] = []
    prepTime: Optional[int] = None  # Mongoose schema defines as Number
    sizes: List[ProductSizeResponse] = []
    toppings: List[ProductToppingResponse] = []
    avgRating: float = 0.0  # Mongoose schema defines as Number, default 0
    location: Optional[ProductLocationResponse] = None  # Nested location object
    createdAt: datetime = Field(..., alias="createdAt")
    updatedAt: datetime = Field(..., alias="updatedAt")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_schema_extra = {}


class VideoResponse(BaseModel):
    id: PyObjectId = Field(..., alias="_id")
    userId: UserResponse
    targetUserId: Optional[UserResponse] = None
    url: str
    filename: str
    direction: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    thumbnail: Optional[str] = None
    hashtags: List[str] = []
    products: List[ProductResponse] = []
    visibility: str  # Added visibility
    views: int = 0
    manifestUrl: Optional[str] = None
    status: str
    duration: float = 0.0
    createdAt: datetime = Field(..., alias="createdAt")
    updatedAt: datetime = Field(..., alias="updatedAt")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_schema_extra = {}
