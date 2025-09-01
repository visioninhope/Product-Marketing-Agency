# Product Marketing Agency Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Agent System](#agent-system)
7. [Configuration](#configuration)
8. [Advanced Usage](#advanced-usage)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

## Overview

Product Marketing Agency is a comprehensive multi-agent system that automates the creation of professional product marketing images using AI. The system leverages the swarms framework to orchestrate specialized agents, each responsible for generating specific types of marketing visuals that meet modern e-commerce and advertising standards.

### Key Features

- **10 Specialized Image Types**: Master shots, flat lays, macro details, lifestyle shots, and more
- **Multi-Agent Architecture**: Dedicated agents for each image type with specialized prompts
- **Rich Interactive UI**: Beautiful terminal interface with progress tracking and live updates
- **Product Profile Management**: JSON-based storage for product information and campaign data
- **Batch Processing**: Generate multiple image types in coordinated campaigns
- **Base64 Image Processing**: Automatic handling of image encoding/decoding
- **Campaign Reporting**: Comprehensive reports with metrics and performance data
- **Error Recovery**: Robust retry mechanisms and fallback strategies

## Architecture

The Product Marketing Agency consists of the following components:

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ Master Product      │    │ Flat Lay            │    │ Macro Detail        │
│ Shot Agent          │    │ Agent               │    │ Agent               │
└─────────────────┘        └─────────────────┘        └─────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ Color Variations    │    │ Size Comparison     │    │ Model Composite     │
│ Agent               │    │ Agent               │    │ Agent               │
└─────────────────┘        └─────────────────┘        └─────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ Lifestyle Action    │    │ UGC Style           │    │ Banner Design       │
│ Agent               │    │ Agent               │    │ Agent               │
└─────────────────┘        └─────────────────┘        └─────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────────────────────────────────┐
│ Shop the Look       │    │ Marketing Campaign Orchestrator                │
│ Agent               │    │ (Coordinates all agents and manages workflow)  │
└─────────────────┘        └─────────────────────────────────────────────────┘
```

### Workflow Process

1. **Product Profile Creation**: Define product details, features, and objectives
2. **Agent Initialization**: Load specialized agents based on selected image types
3. **Image Generation**: Each agent generates its specific image type
4. **Quality Control**: Validate generated images and apply retry logic if needed
5. **Post-Processing**: Convert base64 to image files and organize outputs
6. **Campaign Reporting**: Generate comprehensive reports with all campaign data
7. **Storage Management**: Save profiles, images, and reports in organized structure

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (or other supported LLM provider)
- 4GB+ RAM recommended for image processing
- Terminal with Unicode support for Rich UI

### Install from PyPI

```bash
pip install product-marketing-agency
```

### Install from Source

```bash
git clone https://github.com/The-Swarm-Corporation/Product-Marketing-Agency.git
cd Product-Marketing-Agency
pip install -e .
```

### Install with Poetry

```bash
git clone https://github.com/The-Swarm-Corporation/Product-Marketing-Agency.git
cd Product-Marketing-Agency
poetry install
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
# Required - Choose one or more LLM providers
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_google_api_key_here

# Optional - Configuration
MODEL_NAME=gpt-4o  # Default model to use
MAX_RETRIES=3  # Maximum retry attempts
VERBOSE=false  # Enable detailed logging
OUTPUT_DIR=./output  # Custom output directory
```

## Quick Start

### Basic Usage

```python
from product_marketing_agency import ProductMarketingAgency

# Initialize the agency
agency = ProductMarketingAgency(
    model_name="gpt-4o",
    max_retries=3
)

# Create product profile
product_data = {
    "product_name": "Premium Wireless Headphones",
    "category": "Electronics",
    "key_features": ["Noise cancellation", "40-hour battery", "Premium sound"],
    "accessories": ["Charging cable", "Carrying case", "Audio cable"],
    "objectives": ["Showcase premium quality", "Highlight features"],
    "suggested_image_types": [1, 2, 3, 7, 9]
}

agency.create_product_profile(product_data)

# Run marketing campaign
results = agency.run_campaign()

print(f"Generated {results['images_generated']} marketing images")
```

### Interactive Mode

```python
# Run in interactive mode with menu
agency = ProductMarketingAgency(interactive=True)
agency.run_interactive()
```

## API Reference

### ProductMarketingAgency Class

#### Constructor

```python
ProductMarketingAgency(
    model_name: str = "gpt-4o",
    max_retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    base_path: str = "./output",
    verbose: bool = False,
    interactive: bool = False
)
```

**Parameters:**
- `model_name`: LLM model to use for all agents
- `max_retries`: Maximum retry attempts for failed operations
- `temperature`: LLM temperature for creativity control
- `max_tokens`: Maximum tokens per generation
- `base_path`: Base directory for all outputs
- `verbose`: Enable detailed logging
- `interactive`: Enable interactive menu mode

#### Methods

##### `create_product_profile(product_data: Dict) -> bool`

Create a new product profile for marketing campaign.

**Parameters:**
- `product_data`: Dictionary containing product information

**Returns:**
- `bool`: Success status

##### `run_campaign() -> Dict[str, Any]`

Execute complete marketing campaign for current product.

**Returns:**
- `Dict`: Campaign results with metrics and image details

##### `generate_single_image(image_type: int, custom_prompt: str = None) -> Dict`

Generate a single marketing image of specified type.

**Parameters:**
- `image_type`: ImageType enum value (1-10)
- `custom_prompt`: Optional custom prompt override

**Returns:**
- `Dict`: Image generation results

##### `save_campaign_report() -> str`

Save comprehensive campaign report to JSON file.

**Returns:**
- `str`: Path to saved report file

### ProductProfile Class

```python
@dataclass
class ProductProfile:
    product_name: str
    category: str
    key_features: List[str]
    accessories: List[str]
    objectives: List[str]
    suggested_image_types: List[int]
    product_id: str = None
    master_image_path: Optional[str] = None
    timestamp: str = None
```

### ImageType Enum

```python
class ImageType(Enum):
    MASTER_PRODUCT_SHOT = 1
    WHATS_IN_THE_BOX_FLAT_LAY = 2
    EXTREME_MACRO_DETAIL = 3
    COLOR_STYLE_VARIATIONS = 4
    ON_FOOT_SIZE_COMPARISONS = 5
    ADD_A_MODEL_TWO_IMAGE_COMPOSITE = 6
    LIFESTYLE_ACTION_SHOT = 7
    UGC_STYLE_PHOTOS = 8
    NEGATIVE_SPACE_BANNER = 9
    SHOP_THE_LOOK_FLAT_LAY = 10
```

## Agent System

### Master Product Shot Agent

Generates hero product images with:
- Clean, professional backgrounds
- Optimal lighting and angles
- Focus on product presentation
- E-commerce ready formatting

### What's in the Box Flat Lay Agent

Creates comprehensive flat lay images showing:
- All included items and accessories
- Organized, aesthetic arrangement
- Clear visibility of each component
- Unboxing experience visualization

### Extreme Macro Detail Agent

Produces ultra-close-up shots highlighting:
- Material textures and quality
- Craftsmanship details
- Unique product features
- Premium finishing touches

### Color/Style Variations Agent

Generates images showing:
- All available colors/styles
- Side-by-side comparisons
- Consistent lighting across variants
- Clear differentiation between options

### On-Foot/Size Comparisons Agent

Creates practical demonstration images:
- Product in use/worn
- Size reference comparisons
- Real-world context
- Scale visualization

### Add a Model Two-Image Composite Agent

Produces lifestyle compositions with:
- Human element integration
- Natural usage scenarios
- Emotional connection
- Brand storytelling

### Lifestyle Action Shot Agent

Generates dynamic images featuring:
- Product in action/motion
- Real-world usage scenarios
- Environmental context
- Energy and movement

### UGC Style Photos Agent

Creates authentic-looking content:
- User-generated content aesthetic
- Casual, relatable styling
- Social media ready format
- Natural, unposed feel

### Negative Space Banner Agent

Designs marketing banners with:
- Strategic negative space
- Text overlay areas
- Clean composition
- Advertising-ready layouts

### Shop the Look Flat Lay Agent

Creates styled compositions showing:
- Complete outfit/set coordination
- Complementary product pairings
- Lifestyle aspiration
- Cross-selling opportunities

## Configuration

### Environment Variables

```bash
# LLM Provider Configuration
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GEMINI_API_KEY=your_key
MODEL_NAME=gpt-4o

# Generation Settings
TEMPERATURE=0.1
MAX_TOKENS=4000
MAX_RETRIES=3

# Output Configuration
OUTPUT_DIR=./output
IMAGE_FORMAT=png
IMAGE_QUALITY=high

# UI Settings
RICH_TRACEBACK=True
CONSOLE_WIDTH=120
VERBOSE=false
```

### Model Selection

Supported models:
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- **Google**: `gemini-pro`, `gemini-1.5-pro`, `gemini-2.0-flash`

### Image Generation Parameters

```python
# Image generation settings
IMAGE_CONFIG = {
    "resolution": "1024x1024",
    "quality": "high",
    "style": "photorealistic",
    "format": "png",
    "compression": 0.9
}
```

## Advanced Usage

### Custom Agent Configuration

```python
# Customize individual agent prompts
agency = ProductMarketingAgency()
agency.agents[0].system_prompt = custom_prompt  # Modify master shot agent
```

### Batch Processing Multiple Products

```python
products = [
    {"product_name": "Product A", ...},
    {"product_name": "Product B", ...},
    {"product_name": "Product C", ...}
]

results = []
for product in products:
    agency.create_product_profile(product)
    campaign_result = agency.run_campaign()
    results.append(campaign_result)
    agency.save_campaign_report()
```

### Custom Image Type Selection

```python
# Generate only specific image types
selected_types = [
    ImageType.MASTER_PRODUCT_SHOT.value,
    ImageType.LIFESTYLE_ACTION_SHOT.value,
    ImageType.NEGATIVE_SPACE_BANNER.value
]

for image_type in selected_types:
    result = agency.generate_single_image(image_type)
    print(f"Generated: {ImageType(image_type).name}")
```

### Campaign Analytics

```python
results = agency.run_campaign()

# Analyze performance metrics
metrics = results['execution_metrics']
print(f"Total generation time: {metrics['total_time']:.2f}s")
print(f"Average time per image: {metrics['avg_time_per_image']:.2f}s")
print(f"Success rate: {metrics['success_rate']}%")

# Get detailed agent performance
for agent_name, timing in metrics['agent_timings'].items():
    print(f"{agent_name}: {timing:.2f}s")
```

### Error Handling and Recovery

```python
try:
    results = agency.run_campaign()
except Exception as e:
    # Automatic retry with fallback
    agency.max_retries = 5
    results = agency.run_campaign_with_fallback()
    
    # Check partial results
    if results['partial_success']:
        print(f"Completed {results['images_generated']} of {results['total_requested']}")
```

## Examples

### Example 1: E-commerce Product Launch

```python
from product_marketing_agency import ProductMarketingAgency

# Initialize for e-commerce
agency = ProductMarketingAgency(
    model_name="gpt-4o",
    max_retries=3,
    verbose=True
)

# Define product for launch
product = {
    "product_name": "Smart Watch Pro 2024",
    "category": "Wearable Technology",
    "key_features": [
        "Health monitoring",
        "GPS tracking",
        "Water resistant",
        "7-day battery"
    ],
    "accessories": [
        "Magnetic charger",
        "Extra bands",
        "Screen protectors"
    ],
    "objectives": [
        "Highlight smart features",
        "Show lifestyle integration",
        "Demonstrate durability"
    ],
    "suggested_image_types": [1, 2, 3, 4, 7, 9]
}

# Create profile and run campaign
agency.create_product_profile(product)
results = agency.run_campaign()

# Generate additional marketing materials
banner = agency.generate_single_image(
    image_type=9,
    custom_prompt="Minimalist banner with 'Revolutionary Smart Watch' text"
)

print(f"E-commerce assets ready: {results['images_generated']} images generated")
```

### Example 2: Fashion Product Shoot

```python
# Configure for fashion photography
agency = ProductMarketingAgency(
    model_name="gpt-4o",
    temperature=0.3  # Higher creativity for fashion
)

fashion_product = {
    "product_name": "Vintage Leather Jacket",
    "category": "Fashion Apparel",
    "key_features": [
        "Genuine leather",
        "Hand-stitched details",
        "Limited edition",
        "Custom hardware"
    ],
    "accessories": [
        "Care instructions",
        "Dust bag",
        "Authenticity card"
    ],
    "objectives": [
        "Convey luxury and quality",
        "Show versatility",
        "Target fashion enthusiasts"
    ],
    "suggested_image_types": [1, 3, 6, 7, 8, 10]
}

agency.create_product_profile(fashion_product)
results = agency.run_campaign()

# Focus on lifestyle and styling
for style_type in [7, 8, 10]:  # Lifestyle, UGC, Shop the Look
    agency.generate_single_image(style_type)
```

### Example 3: Multi-Product Campaign

```python
# Batch process product line
product_line = [
    {
        "product_name": "Yoga Mat Premium",
        "category": "Fitness",
        "suggested_image_types": [1, 2, 7]
    },
    {
        "product_name": "Resistance Bands Set",
        "category": "Fitness",
        "suggested_image_types": [1, 2, 4]
    },
    {
        "product_name": "Foam Roller Pro",
        "category": "Fitness",
        "suggested_image_types": [1, 3, 7]
    }
]

campaign_results = []

for product in product_line:
    print(f"\nProcessing: {product['product_name']}")
    
    # Add common features
    product.update({
        "key_features": ["Premium quality", "Durable", "Eco-friendly"],
        "accessories": ["Instructions", "Carrying bag"],
        "objectives": ["Show quality", "Demonstrate use"]
    })
    
    agency.create_product_profile(product)
    result = agency.run_campaign()
    campaign_results.append(result)
    
    # Save individual report
    report_path = agency.save_campaign_report()
    print(f"Report saved: {report_path}")

# Summary statistics
total_images = sum(r['images_generated'] for r in campaign_results)
total_time = sum(r['total_time'] for r in campaign_results)
print(f"\nCampaign complete: {total_images} images in {total_time:.2f}s")
```

## Troubleshooting

### Common Issues

#### 1. API Rate Limits

**Problem**: `RateLimitError` from OpenAI or other providers

**Solution**:
```python
# Add delays between generations
import time

agency = ProductMarketingAgency(
    model_name="gpt-4o-mini",  # Use faster model
    max_retries=5  # Increase retries
)

# The framework includes automatic retry with exponential backoff
```

#### 2. Image Generation Failures

**Problem**: Images not generating or saving correctly

**Solution**:
```python
# Enable verbose logging
agency = ProductMarketingAgency(verbose=True)

# Check for base64 decoding issues
try:
    result = agency.generate_single_image(1)
    if not result['success']:
        print(f"Error details: {result.get('error')}")
        print(f"Raw response: {result.get('raw_response')[:200]}")
except Exception as e:
    print(f"Generation error: {e}")
```

#### 3. Memory Issues with Large Campaigns

**Problem**: Memory usage high with multiple images

**Solution**:
```python
# Process in smaller batches
def process_in_batches(image_types, batch_size=3):
    for i in range(0, len(image_types), batch_size):
        batch = image_types[i:i+batch_size]
        for img_type in batch:
            agency.generate_single_image(img_type)
        
        # Clear memory between batches
        import gc
        gc.collect()
```

#### 4. JSON Parsing Errors

**Problem**: Agent responses not parsing correctly

**Solution**:
```python
# The framework includes robust JSON extraction
# If issues persist, check agent responses:
agency.verbose = True  # Enable logging
result = agency.generate_single_image(1)

# Manual response inspection
if not result['success']:
    print("Raw agent response:")
    print(result.get('raw_response'))
```

### Performance Optimization

#### 1. Model Selection for Speed

```python
# Fast configuration for prototyping
fast_agency = ProductMarketingAgency(
    model_name="gpt-4o-mini",  # Faster model
    max_tokens=2000,  # Reduced tokens
    temperature=0.1  # More deterministic
)

# High quality for production
production_agency = ProductMarketingAgency(
    model_name="gpt-4o",
    max_tokens=4000,
    temperature=0.3
)
```

#### 2. Parallel Processing

```python
import concurrent.futures

def generate_image_parallel(image_type):
    return agency.generate_single_image(image_type)

# Generate multiple images in parallel
image_types = [1, 2, 3, 4, 5]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(generate_image_parallel, image_types))
```

#### 3. Caching and Reuse

```python
# Cache product profiles
import json

def save_profile_cache(profile, filename="profile_cache.json"):
    with open(filename, 'w') as f:
        json.dump(profile, f)

def load_profile_cache(filename="profile_cache.json"):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Reuse cached profiles
cached = load_profile_cache()
if cached:
    agency.current_product = cached
else:
    agency.create_product_profile(product_data)
    save_profile_cache(agency.current_product)
```

### Debugging Tips

1. **Enable Verbose Logging**:
   ```python
   agency = ProductMarketingAgency(verbose=True)
   ```

2. **Check Agent States**:
   ```python
   # Inspect individual agent configuration
   for i, agent in enumerate(agency.agents):
       print(f"Agent {i}: {agent.agent_name}")
       print(f"Model: {agent.llm.model_name}")
   ```

3. **Monitor Output Directory**:
   ```python
   import os
   output_dir = agency.base_path
   
   # List all generated files
   for root, dirs, files in os.walk(output_dir):
       for file in files:
           print(os.path.join(root, file))
   ```

4. **Validate Image Files**:
   ```python
   from PIL import Image
   
   def validate_image(image_path):
       try:
           img = Image.open(image_path)
           img.verify()
           return True
       except Exception as e:
           print(f"Invalid image {image_path}: {e}")
           return False
   ```

## Support and Resources

- **GitHub Issues**: [Report bugs and request features](https://github.com/The-Swarm-Corporation/Product-Marketing-Agency/issues)
- **Discord Community**: [Join our Discord](https://discord.gg/swarms-999382051935506503)
- **Documentation**: [Full documentation](https://github.com/The-Swarm-Corporation/Product-Marketing-Agency/blob/main/docs/README.md)
- **Examples**: [Example notebooks and scripts](https://github.com/The-Swarm-Corporation/Product-Marketing-Agency/tree/main/examples)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/The-Swarm-Corporation/Product-Marketing-Agency/blob/main/LICENSE) file for details.

## Citation

If you use Product Marketing Agency in your work, please cite:

```bibtex
@software{product_marketing_agency,
    title={Product Marketing Agency: Multi-Agent System for Marketing Image Generation},
    author={The Swarm Corporation},
    year={2024},
    url={https://github.com/The-Swarm-Corporation/Product-Marketing-Agency}
}
```