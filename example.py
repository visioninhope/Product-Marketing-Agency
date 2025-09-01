import json
from product_marketing_agency.main import ProductMarketingAgency

# Initialize the Product Marketing Agency
agency = ProductMarketingAgency(
    model_name="gpt-4o",  # Or "gpt-4o-mini" for faster/cheaper option
    max_retries=2,  # Reduced retries for example run
    verbose=False,  # Set to True for detailed logs
)

# Define product details for marketing campaign
product_data = {
    "product_name": "CloudWalk Pro Running Shoes",
    "category": "Athletic Footwear",
    "key_features": [
        "Ultra-lightweight mesh upper",
        "CloudFoam cushioning technology",
        "Non-slip rubber outsole",
        "Breathable moisture-wicking lining"
    ],
    "accessories": [
        "Extra laces (white and black)",
        "Shoe care kit",
        "Travel shoe bag",
        "Performance insoles"
    ],
    "objectives": [
        "Showcase product versatility",
        "Highlight comfort features",
        "Appeal to fitness enthusiasts",
        "Demonstrate premium quality"
    ],
    "suggested_image_types": [1, 2, 3, 7, 9]  # Master shot, flat lay, detail, UGC, banner
}

# Create product profile
print("\n--- Creating Product Profile ---")
profile_created = agency.create_product_profile(product_data)

if profile_created:
    print(f"✓ Product profile created: {product_data['product_name']}")
    print(f"  Product ID: {agency.current_product['product_id']}")
    print(f"  Category: {agency.current_product['category']}")
    print(f"  Features: {len(agency.current_product['key_features'])} key features")
    print(f"  Suggested images: {agency.current_product['suggested_image_types']}")
else:
    print("✗ Failed to create product profile")
    exit(1)

# Generate marketing images
print("\n--- Generating Marketing Images ---")
results = agency.run_campaign()

# Output the results
print("\n--- Campaign Results ---")
if results["success"]:
    print(f"✓ Campaign completed successfully!")
    print(f"  Images generated: {results['images_generated']}")
    print(f"  Total time: {results['total_time']:.2f} seconds")
    
    print("\n--- Generated Images ---")
    for image_info in results["image_details"]:
        print(f"  • {image_info['type']}:")
        print(f"    Status: {image_info['status']}")
        if image_info['status'] == 'success':
            print(f"    Path: {image_info['path']}")
            print(f"    Size: {image_info.get('size', 'N/A')}")
        print("-" * 30)
    
    print("\n--- Campaign Metrics ---")
    print(json.dumps(results["execution_metrics"], indent=2))
else:
    print(f"✗ Campaign failed: {results.get('error', 'Unknown error')}")

# Save campaign report (optional)
try:
    agency.save_campaign_report()
    print("\n✓ Campaign report saved to output/campaign_reports/")
except Exception as e:
    print(f"\n✗ Error saving report: {e}")

# Example of generating a single image type
print("\n--- Generating Single Image Example ---")
single_image = agency.generate_single_image(
    image_type=1,  # Master Product Shot
    custom_prompt="Premium running shoes on white background with dramatic lighting"
)

if single_image["success"]:
    print(f"✓ Single image generated: {single_image['image_path']}")
else:
    print(f"✗ Failed to generate image: {single_image.get('error', 'Unknown error')}")