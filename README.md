# Product Marketing Agency

![Product Marketing Agency](assets/pms.png)

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/swarms-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

A comprehensive multi-agent system for generating professional product marketing images and content using the Swarms framework. Create stunning marketing materials with AI-powered specialized agents for different image types and marketing objectives.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Framework](https://img.shields.io/badge/Framework-Swarms-orange.svg)

## Overview

The Product Marketing Agency is an advanced AI-driven system that orchestrates multiple specialized agents to create comprehensive marketing materials for products. Each agent is optimized for specific image types and marketing objectives, ensuring professional-quality output tailored to your product's unique characteristics.

## Key Features

### 🎯 10 Specialized Marketing Image Types
- **Master Product Shot**: Hero images showcasing the main product
- **What's in the Box Flat Lay**: Unboxing and contents display
- **Extreme Macro Detail**: Close-up shots highlighting craftsmanship
- **Color/Style Variations**: Product variants and options
- **On-Foot Size Comparisons**: Scale and sizing demonstrations
- **Add a Model Two-Image Composite**: Lifestyle modeling shots
- **Lifestyle Action Shot**: Products in real-world usage
- **UGC Style Photos**: User-generated content aesthetic
- **Negative Space Banner**: Clean promotional banners
- **Shop the Look Flat Lay**: Complete styling and accessory layouts

### 🤖 Multi-Agent Architecture
- Specialized agents for each image type
- Intelligent product analysis and image type recommendations
- Coordinated multi-agent workflows
- Agent performance tracking and optimization

### 📊 Product Profile Management
- JSON-based product data storage
- Comprehensive product categorization
- Feature and accessory tracking
- Marketing objective alignment
- Batch processing capabilities

### 🎨 Rich Interactive Interface
- Enhanced terminal UI with Rich library
- Interactive menu system
- Real-time progress tracking
- Campaign statistics and reporting
- Visual feedback and status updates

### 🔄 Advanced Processing Features
- Base64 image processing and conversion
- Intelligent image type suggestions based on product category
- Batch processing for multiple products
- Campaign orchestration and management
- Error handling and recovery mechanisms

## Installation

### Prerequisites
- Python 3.10 or higher
- Poetry (recommended) or pip
- API key for supported LLM providers (OpenAI, Gemini, etc.)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/The-Swarm-Corporation/Product-Marketing-Agency.git
cd Product-Marketing-Agency
```

2. **Install dependencies:**
```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
# or
export GEMINI_API_KEY="your-gemini-api-key"
```

4. **Run the application:**
```bash
poetry run python product_marketing_agency/main.py
# or
python product_marketing_agency/main.py
```

## Quick Start

### Creating Your First Marketing Campaign

1. **Launch the application** and select "Create New Marketing Campaign"

2. **Define your product profile:**
```json
{
  "product_name": "UltraComfort Running Shoes",
  "category": "footwear",
  "key_features": ["Lightweight", "Breathable mesh", "Responsive cushioning"],
  "accessories": ["Extra laces", "Shoe bag", "Insoles"],
  "objectives": ["Increase sales", "Highlight comfort features", "Target fitness enthusiasts"]
}
```

3. **Select image types** - The system will intelligently suggest optimal image types based on your product category

4. **Review and generate** - The specialized agents will create professional marketing materials tailored to your specifications

### Batch Processing Multiple Products

For processing multiple products simultaneously:

1. Select "Batch Process Multiple Products" from the main menu
2. Provide product data for multiple items
3. Choose default image types or let the system auto-suggest
4. Monitor progress through the Rich UI interface

## Architecture

### Agent Specialization

Each marketing image type has a dedicated agent with specialized knowledge:

```python
class ImageType(Enum):
    MASTER_PRODUCT_SHOT = 1
    WHATS_IN_THE_BOX_FLAT_LAY = 2
    EXTREME_MACRO_DETAIL = 3
    # ... and 7 more specialized types
```

### Product Profile Structure

```python
@dataclass
class ProductProfile:
    product_name: str
    category: str
    key_features: List[str]
    accessories: List[str]
    objectives: List[str]
    suggested_image_types: List[int]
```

### Multi-Agent Workflow

1. **Product Analysis**: Initial agent analyzes product characteristics
2. **Strategy Planning**: Marketing strategy agent determines optimal approach
3. **Specialized Generation**: Individual agents create type-specific content
4. **Quality Assurance**: Review and refinement process
5. **Output Optimization**: Final formatting and delivery

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes* |
| `GEMINI_API_KEY` | Google Gemini API key | Yes* |
| `PMA_OUTPUT_DIR` | Custom output directory | No |
| `PMA_LOG_LEVEL` | Logging level (DEBUG, INFO, WARN, ERROR) | No |

*At least one API key is required

### Directory Structure

```
Product-Marketing-Agency/
├── product_marketing_agency/
│   ├── main.py              # Main application
│   ├── data/                # Product profiles storage
│   ├── output/              # Generated marketing materials
│   └── logs/                # Application logs
├── README.md
├── pyproject.toml
└── LICENSE
```

## Usage Examples

### Creating a Product Profile

```python
# Interactive mode
agency = ProductMarketingAgency()
product_profile = agency.create_product_profile({
    "product_name": "Smart Wireless Headphones",
    "category": "electronics",
    "key_features": ["Noise cancellation", "35-hour battery", "Quick charge"],
    "accessories": ["Charging cable", "Carrying case", "Audio adapter"],
    "objectives": ["Showcase premium build", "Highlight battery life", "Target audiophiles"]
})
```

### Batch Processing

```python
# Process multiple products
products_data = [
    {"product_name": "Product A", "category": "electronics", ...},
    {"product_name": "Product B", "category": "fashion", ...},
]

agency.batch_generate_marketing_content(
    products_data, 
    default_image_types=[ImageType.MASTER_PRODUCT_SHOT, ImageType.LIFESTYLE_ACTION_SHOT]
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the codebase.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `poetry install --with dev`
4. Make your changes and add tests
5. Run tests: `pytest`
6. Submit a pull request

## Roadmap

- [ ] Integration with additional AI providers (Claude, LLaMA)
- [ ] Advanced image editing and post-processing capabilities
- [ ] Social media platform optimization
- [ ] A/B testing framework for marketing materials
- [ ] API endpoints for external integrations
- [ ] Web-based dashboard interface
- [ ] Advanced analytics and performance tracking

## Support

- **Documentation**: [Project Wiki](https://github.com/The-Swarm-Corporation/Product-Marketing-Agency/wiki)
- **Issues**: [GitHub Issues](https://github.com/The-Swarm-Corporation/Product-Marketing-Agency/issues)
- **Discussions**: [GitHub Discussions](https://github.com/The-Swarm-Corporation/Product-Marketing-Agency/discussions)
- **Community**: [Discord Server](https://discord.gg/swarms)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with the [Swarms Framework](https://github.com/kyegomez/swarms)
- UI powered by [Rich](https://github.com/Textualize/rich)
- Inspired by modern marketing automation needs

---

**Made with ❤️ by The Swarm Corporation**
