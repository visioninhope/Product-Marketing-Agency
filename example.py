import os

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from dotenv import load_dotenv

from product_marketing_agency.main import (
    ImageType,
    ProductMarketingAgency,
    display_welcome_banner,
    get_product_information_interactive,
)

load_dotenv()

console = Console()


def main_menu():
    """Main application menu"""

    # Initialize the agency
    agency = ProductMarketingAgency()

    while True:
        # Create main menu with Rich table
        menu_table = Table(
            title="[bold magenta]MAIN MENU[/bold magenta]"
        )
        menu_table.add_column("Option", style="cyan bold", width=8)
        menu_table.add_column("Description", style="white")

        menu_table.add_row("1", "Create New Marketing Campaign")
        menu_table.add_row("2", "Batch Process Multiple Products")
        menu_table.add_row("3", "Load Existing Product Profile")
        menu_table.add_row("4", "View Campaign Statistics")
        menu_table.add_row("5", "Set Master Reference Image")
        menu_table.add_row("6", "Help & Documentation")
        menu_table.add_row("7", "Exit")

        console.print("\n")
        console.print(menu_table)

        choice = Prompt.ask(
            "\n[bold cyan]Select option[/bold cyan]",
            choices=["1", "2", "3", "4", "5", "6", "7"],
        )

        try:
            if choice == "1":
                # Single product campaign
                product_data = get_product_information_interactive()

                # Extract custom requirements and master image from bulk input
                custom_req = product_data.pop(
                    "custom_requirements", ""
                )
                master_img_path = product_data.pop(
                    "master_image_path", ""
                )

                # Validate master image path if provided
                master_img = None
                if master_img_path and os.path.exists(
                    master_img_path
                ):
                    master_img = master_img_path
                    print(f"✅ Master image found: {master_img_path}")
                elif master_img_path:
                    print(
                        f"⚠️  Master image not found: {master_img_path}"
                    )

                results = agency.run_marketing_campaign(
                    product_data=product_data,
                    custom_requirements=custom_req,
                    master_image_path=master_img,
                    interactive=True,
                )

                success_panel = Panel(
                    f"Campaign completed! Generated {len(results)} marketing assets.",
                    title="[bold green]Success![/bold green]",
                    border_style="green",
                )
                console.print("\n")
                console.print(success_panel)

            elif choice == "2":
                # Batch processing
                batch_panel = Panel(
                    "[bold yellow]BATCH PROCESSING MODE[/bold yellow]\nThis feature would load products from a JSON file or database.\nFor demo purposes, using sample data...",
                    border_style="yellow",
                )
                console.print("\n")
                console.print(batch_panel)

                # Demo batch data
                sample_products = [
                    {
                        "name": "EcoSmart Water Bottle",
                        "category": "Sustainability",
                        "description": (
                            "Smart water bottle with temperature control"
                        ),
                        "features": [
                            "Temperature control",
                            "App connectivity",
                            "Leak-proof",
                        ],
                        "target_audience": (
                            "Health-conscious professionals"
                        ),
                        "brand_colors": ["#22c55e", "#ffffff"],
                        "price_range": "$49-79",
                        "usp": [
                            "24hr temperature retention",
                            "Smart hydration tracking",
                        ],
                    },
                    {
                        "name": "PowerDesk Pro",
                        "category": "Office Equipment",
                        "description": (
                            "Height-adjustable standing desk with wireless charging"
                        ),
                        "features": [
                            "Height adjustment",
                            "Wireless charging pad",
                            "Cable management",
                        ],
                        "target_audience": (
                            "Remote workers and professionals"
                        ),
                        "brand_colors": ["#1f2937", "#f59e0b"],
                        "price_range": "$299-499",
                        "usp": [
                            "Wireless charging integration",
                            "Memory presets",
                        ],
                    },
                ]

                agency.batch_process_products(sample_products)
                completion_panel = Panel(
                    f"Batch processing completed! Processed {len(sample_products)} products.",
                    title="[bold green]Batch Complete![/bold green]",
                    border_style="green",
                )
                console.print("\n")
                console.print(completion_panel)

            elif choice == "3":
                # Load existing profile
                profiles = agency.list_available_profiles()
                if profiles:
                    profiles_text = (
                        f"Available Profiles: {', '.join(profiles)}"
                    )
                    console.print(f"\n[cyan]{profiles_text}[/cyan]")
                    profile_id = Prompt.ask(
                        "Enter profile ID to load"
                    )

                    profile = agency.load_product_profile(profile_id)
                    if profile:
                        console.print(
                            f"\n[bold green]✓ Loaded profile: {profile.product_name}[/bold green]"
                        )
                        # Continue with campaign using existing profile...
                    else:
                        console.print(
                            "[red]❌ Profile not found.[/red]"
                        )
                else:
                    console.print(
                        "\n[yellow]⚠️ No existing profiles found.[/yellow]"
                    )

            elif choice == "4":
                # Statistics
                stats = agency.get_campaign_statistics()
                # Create statistics table
                stats_table = Table(
                    title="[bold cyan]📊 CAMPAIGN STATISTICS[/bold cyan]"
                )
                stats_table.add_column("Metric", style="green bold")
                stats_table.add_column("Value", style="white")

                stats_table.add_row(
                    "Total Products", str(stats["total_products"])
                )
                stats_table.add_row(
                    "Total Jobs", str(stats["total_jobs"])
                )
                stats_table.add_row(
                    "Completed Jobs", str(stats["completed_jobs"])
                )
                stats_table.add_row(
                    "Save Directory", stats["save_directory"]
                )

                console.print("\n")
                console.print(stats_table)

                if stats["image_types_generated"]:
                    types_table = Table(
                        title="[bold yellow]Image Types Generated[/bold yellow]"
                    )
                    types_table.add_column("Type", style="magenta")
                    types_table.add_column("Count", style="cyan")

                    for img_type, count in stats[
                        "image_types_generated"
                    ].items():
                        types_table.add_row(
                            img_type.replace("_", " ").title(),
                            str(count),
                        )

                    console.print("\n")
                    console.print(types_table)

            elif choice == "5":
                # Set master image
                product_id = input("Product ID: ").strip()
                image_path = input("Master Image Path: ").strip()

                try:
                    agency.set_master_image(product_id, image_path)
                    print(" Master image set successfully.")
                except Exception as e:
                    print(f" Error: {str(e)}")

            elif choice == "6":
                # Help
                print("\n= HELP & DOCUMENTATION")
                print("-" * 40)
                print(
                    "This system creates marketing materials using 6 specialized AI agents:"
                )
                print("1. Orchestrator Agent - Manages workflow")
                print(
                    "2. Product Interpreter Agent - Analyzes products"
                )
                print(
                    "3. Image Type Selector Agent - Guides selection"
                )
                print(
                    "4. Prompt Generator Agent - Creates custom prompts"
                )
                print("5. Image Generation Agent - Generates content")
                print("6. Output Handler Agent - Manages feedback")
                print("\nSupported Image Types:")
                for img_type in ImageType:
                    image_name = img_type.name.replace(
                        "_", " "
                    ).title()
                    print(f" {image_name}")

            elif choice == "7":
                print(
                    "\n= Thank you for using Product Marketing Agency!"
                )
                print(
                    "All generated content is saved in the outputs directory."
                )
                break

            else:
                print("L Invalid option. Please select 1-7.")

        except KeyboardInterrupt:
            print("\n\n= Operation cancelled. Goodbye!")
            break
        except Exception as e:
            print(f"\nL Error: {str(e)}")
            print("Please try again or contact support.")


# ==========================================================================
# Main Application Entry Point
# ==========================================================================

if __name__ == "__main__":
    display_welcome_banner()
    main_menu()
