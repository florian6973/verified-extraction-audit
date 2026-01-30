"""
Build a filter list of first names or last names from multiple faker locales.

This script extracts first names (male, female, and nonbinary) or last names from various
language providers in the faker package and creates a comprehensive filter list.
"""

import os
import sys
from typing import Set, List, Dict
from collections import OrderedDict

try:
    from faker import Faker
    from faker.providers import BaseProvider
except ImportError:
    print("Error: faker package not installed. Please install it with: pip install faker")
    sys.exit(1)


def get_first_names_from_provider(provider_class) -> Set[str]:
    """
    Extract all first names from a faker person provider class.
    
    Args:
        provider_class: The provider class (e.g., faker.providers.person.en_US.Provider)
    
    Returns:
        Set of first names (male, female, and nonbinary if available)
    """
    names = set()
    
    # Try to get male first names
    if hasattr(provider_class, 'first_names_male'):
        first_names_male = provider_class.first_names_male
        if isinstance(first_names_male, (dict, OrderedDict)):
            names.update(first_names_male.keys())
        elif isinstance(first_names_male, (list, tuple)):
            names.update(first_names_male)
    
    # Try to get female first names
    if hasattr(provider_class, 'first_names_female'):
        first_names_female = provider_class.first_names_female
        if isinstance(first_names_female, (dict, OrderedDict)):
            names.update(first_names_female.keys())
        elif isinstance(first_names_female, (list, tuple)):
            names.update(first_names_female)
    
    # Try to get nonbinary first names (if available)
    if hasattr(provider_class, 'first_names_nonbinary'):
        first_names_nonbinary = provider_class.first_names_nonbinary
        if isinstance(first_names_nonbinary, (dict, OrderedDict)):
            names.update(first_names_nonbinary.keys())
        elif isinstance(first_names_nonbinary, (list, tuple)):
            names.update(first_names_nonbinary)
    
    # Fallback: try to get general first_names
    if hasattr(provider_class, 'first_names'):
        first_names = provider_class.first_names
        if isinstance(first_names, (dict, OrderedDict)):
            names.update(first_names.keys())
        elif isinstance(first_names, (list, tuple)):
            names.update(first_names)
    
    return names


def get_last_names_from_provider(provider_class) -> Set[str]:
    """
    Extract all last names from a faker person provider class.
    
    Args:
        provider_class: The provider class (e.g., faker.providers.person.en_US.Provider)
    
    Returns:
        Set of last names
    """
    names = set()
    
    # Try to get last names
    if hasattr(provider_class, 'last_names'):
        last_names = provider_class.last_names
        if isinstance(last_names, (dict, OrderedDict)):
            names.update(last_names.keys())
        elif isinstance(last_names, (list, tuple)):
            names.update(last_names)
    
    return names


def get_names_from_locale(locale: str, name_type: str = 'first') -> Set[str]:
    """
    Get names from a specific locale by accessing the provider directly.
    
    Args:
        locale: Locale string (e.g., 'en_US', 'es_ES', 'pt_BR')
        name_type: 'first' or 'last' (default: 'first')
    
    Returns:
        Set of names from that locale
    """
    names = set()
    
    try:
        # Create a faker instance for this locale
        fake = Faker(locale)
        
        # Try to access the provider directly
        person_provider = None
        for provider in fake.providers:
            # Check if this provider has name-related attributes
            if name_type == 'first':
                if (hasattr(provider, 'first_names_male') or 
                    hasattr(provider, 'first_names_female') or
                    hasattr(provider, 'first_names') or
                    hasattr(provider.__class__, 'first_names_male') or
                    hasattr(provider.__class__, 'first_names_female')):
                    person_provider = provider
                    break
            else:  # last names
                if (hasattr(provider, 'last_names') or
                    hasattr(provider.__class__, 'last_names')):
                    person_provider = provider
                    break
        
        if person_provider:
            # Try both instance and class attributes
            if name_type == 'first':
                names = get_first_names_from_provider(person_provider.__class__)
                if not names:
                    # Try instance attributes
                    if hasattr(person_provider, 'first_names_male'):
                        male_names = person_provider.first_names_male
                        if isinstance(male_names, (dict, OrderedDict)):
                            names.update(male_names.keys())
                    if hasattr(person_provider, 'first_names_female'):
                        female_names = person_provider.first_names_female
                        if isinstance(female_names, (dict, OrderedDict)):
                            names.update(female_names.keys())
            else:  # last names
                names = get_last_names_from_provider(person_provider.__class__)
                if not names:
                    # Try instance attributes
                    if hasattr(person_provider, 'last_names'):
                        last_names = person_provider.last_names
                        if isinstance(last_names, (dict, OrderedDict)):
                            names.update(last_names.keys())
        
        # Fallback: generate many names and collect unique ones
        # This is less reliable but works when we can't access the data directly
        if not names:
            print(f"  Using generation method for {locale} (may be incomplete)")
            generated_names = set()
            # Generate a large sample to get most names
            for _ in range(50000):
                try:
                    if name_type == 'first':
                        if hasattr(fake, 'first_name_male'):
                            generated_names.add(fake.first_name_male())
                        if hasattr(fake, 'first_name_female'):
                            generated_names.add(fake.first_name_female())
                        if hasattr(fake, 'first_name'):
                            generated_names.add(fake.first_name())
                    else:  # last names
                        if hasattr(fake, 'last_name'):
                            generated_names.add(fake.last_name())
                except:
                    pass
            names = generated_names
            
    except Exception as e:
        print(f"  Error processing locale {locale}: {e}")
    
    return names


def get_names_via_import(locale: str, name_type: str = 'first') -> Set[str]:
    """
    Try to import the provider module directly and extract names.
    
    Args:
        locale: Locale string (e.g., 'en_US', 'es_ES')
        name_type: 'first' or 'last' (default: 'first')
    
    Returns:
        Set of names from that locale
    """
    names = set()
    
    # Map locale to module path
    locale_mapping = {
        'en_US': 'faker.providers.person.en_US',
        'en_GB': 'faker.providers.person.en_GB',
        'es_ES': 'faker.providers.person.es_ES',
        'es_MX': 'faker.providers.person.es_MX',
        'pt_BR': 'faker.providers.person.pt_BR',
        'pt_PT': 'faker.providers.person.pt_PT',
        'fr_FR': 'faker.providers.person.fr_FR',
        'ru_RU': 'faker.providers.person.ru_RU',
        'ar_AA': 'faker.providers.person.ar_AA',
        'ar_EG': 'faker.providers.person.ar_EG',
        'ar_SA': 'faker.providers.person.ar_SA',
        'zh_CN': 'faker.providers.person.zh_CN',
        'zh_TW': 'faker.providers.person.zh_TW',
        'vi_VN': 'faker.providers.person.vi_VN',
        'ja_JP': 'faker.providers.person.ja_JP',
        'ko_KR': 'faker.providers.person.ko_KR',
        'de_DE': 'faker.providers.person.de_DE',
        'it_IT': 'faker.providers.person.it_IT',
        'nl_NL': 'faker.providers.person.nl_NL',
        'pl_PL': 'faker.providers.person.pl_PL',
        'tr_TR': 'faker.providers.person.tr_TR',
        'hi_IN': 'faker.providers.person.hi_IN',
        'th_TH': 'faker.providers.person.th_TH',
    }
    
    module_path = locale_mapping.get(locale)
    if not module_path:
        # Try to construct it
        parts = locale.split('_')
        if len(parts) == 2:
            module_path = f"faker.providers.person.{parts[0].lower()}_{parts[1].upper()}"
    
    if module_path:
        try:
            module = __import__(module_path, fromlist=['Provider'])
            provider_class = getattr(module, 'Provider', None)
            if provider_class:
                if name_type == 'first':
                    names = get_first_names_from_provider(provider_class)
                else:  # last names
                    names = get_last_names_from_provider(provider_class)
                if names:
                    return names
        except (ImportError, AttributeError) as e:
            # Try alternative import method
            try:
                # Some locales might use different naming conventions
                parts = locale.split('_')
                if len(parts) == 2:
                    # Try lowercase both parts
                    alt_path = f"faker.providers.person.{parts[0].lower()}_{parts[1].lower()}"
                    module = __import__(alt_path, fromlist=['Provider'])
                    provider_class = getattr(module, 'Provider', None)
                    if provider_class:
                        if name_type == 'first':
                            names = get_first_names_from_provider(provider_class)
                        else:  # last names
                            names = get_last_names_from_provider(provider_class)
            except:
                pass
    
    return names


def discover_available_locales(name_type: str = 'first') -> List[str]:
    """
    Discover available person provider locales in faker.
    
    Args:
        name_type: 'first' or 'last' (default: 'first')
    
    Returns:
        List of available locale strings
    """
    available_locales = []
    
    try:
        # Try to list available locales from faker
        from faker.config import AVAILABLE_LOCALES
        
        # Filter for locales that likely have person providers
        person_locales = []
        for locale in AVAILABLE_LOCALES:
            try:
                fake = Faker(locale)
                # Check if this locale has person provider methods
                if name_type == 'first':
                    if (hasattr(fake, 'first_name') or 
                        hasattr(fake, 'first_name_male') or 
                        hasattr(fake, 'first_name_female')):
                        person_locales.append(locale)
                else:  # last names
                    if hasattr(fake, 'last_name'):
                        person_locales.append(locale)
            except:
                pass
        
        return person_locales
    except:
        # Fallback: return common locales
        return [
            'en_US', 'en_GB', 'es_ES', 'es_MX', 'pt_BR', 'pt_PT',
            'fr_FR', 'ru_RU', 'ar_AA', 'ar_EG', 'ar_SA',
            'zh_CN', 'zh_TW', 'vi_VN', 'ja_JP', 'ko_KR',
            'de_DE', 'it_IT', 'nl_NL', 'pl_PL', 'tr_TR', 'hi_IN', 'th_TH'
        ]


def build_name_filter_list(locales: List[str] = None, name_type: str = 'first') -> Set[str]:
    """
    Build a comprehensive filter list of names from multiple locales.
    
    Args:
        locales: List of locale strings. If None, uses a default set.
        name_type: 'first' or 'last' (default: 'first')
    
    Returns:
        Set of all unique names
    """
    if locales is None:
        # Default locales covering the requested languages
        locales = [
            'en_US',  # English (US)
            'en_GB',  # English (UK)
            'es_ES',  # Spanish (Spain)
            'es_MX',  # Spanish (Mexico)
            'pt_BR',  # Portuguese (Brazil)
            'pt_PT',  # Portuguese (Portugal)
            'fr_FR',  # French
            'ru_RU',  # Russian
            'ar_AA',  # Arabic (generic)
            'ar_EG',  # Arabic (Egypt)
            'ar_SA',  # Arabic (Saudi Arabia)
            'zh_CN',  # Chinese (Simplified)
            'zh_TW',  # Chinese (Traditional)
            'vi_VN',  # Vietnamese
            'ja_JP',  # Japanese
            'ko_KR',  # Korean
            'de_DE',  # German
            'it_IT',  # Italian
            'nl_NL',  # Dutch
            'pl_PL',  # Polish
            'tr_TR',  # Turkish
            'hi_IN',  # Hindi
            'th_TH',  # Thai
        ]
    
    all_names = set()
    
    name_type_label = 'first names' if name_type == 'first' else 'last names'
    print(f"Extracting {name_type_label} from {len(locales)} locales...")
    
    for locale in locales:
        print(f"\nProcessing locale: {locale}")
        
        # Try direct import first (most reliable)
        names = get_names_via_import(locale, name_type)
        
        # Fallback to locale-based generation
        if not names:
            names = get_names_from_locale(locale, name_type)
        
        if names:
            print(f"  Found {len(names)} unique {name_type_label}")
            all_names.update(names)
        else:
            print(f"  Warning: No {name_type_label} found for {locale}")
    
    return all_names


def save_filter_list(names: Set[str], output_path: str = None, name_type: str = 'first'):
    """
    Save the filter list to a file.
    
    Args:
        names: Set of names to save
        output_path: Path to output file. If None, uses default.
        name_type: 'first' or 'last' (default: 'first')
    """
    if output_path is None:
        # Default to same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if name_type == 'first':
            output_path = os.path.join(script_dir, 'faker_first_names_filter.txt')
        else:
            output_path = os.path.join(script_dir, 'faker_last_names_filter.txt')
    
    # Sort names for consistent output
    sorted_names = sorted(names, key=str.lower)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for name in sorted_names:
            f.write(f"{name}\n")
    
    print(f"\nSaved {len(sorted_names)} names to {output_path}")
    
    # Also save as Python set for easy import
    py_output_path = output_path.replace('.txt', '.py')
    with open(py_output_path, 'w', encoding='utf-8') as f:
        name_type_label = 'First' if name_type == 'first' else 'Last'
        f.write(f"# {name_type_label} names filter list from faker package\n")
        f.write("# Generated automatically - do not edit manually\n\n")
        var_name = 'FIRST_NAMES_FILTER' if name_type == 'first' else 'LAST_NAMES_FILTER'
        f.write(f"{var_name} = {{\n")
        for name in sorted_names:
            # Escape quotes in names
            escaped_name = name.replace("'", "\\'").replace('"', '\\"')
            f.write(f"    '{escaped_name}',\n")
        f.write("}\n")
    
    print(f"Saved Python set to {py_output_path}")


def main():
    """Main function to build and save the filter list."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build a filter list of names from faker package locales'
    )
    parser.add_argument(
        '--locales',
        nargs='+',
        help='Specific locales to use (e.g., en_US es_ES pt_BR). If not provided, uses default set.'
    )
    parser.add_argument(
        '--discover',
        action='store_true',
        help='Auto-discover available locales instead of using defaults'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: faker_first_names_filter.txt or faker_last_names_filter.txt in script directory)'
    )
    parser.add_argument(
        '--last-names',
        action='store_true',
        help='Build last names filter instead of first names filter'
    )
    parser.add_argument(
        '--max-display',
        type=int,
        default=None,
        help='Maximum number of sample names to display (default: display all)'
    )
    
    args = parser.parse_args()
    
    name_type = 'last' if args.last_names else 'first'
    name_type_label = 'Last Names' if args.last_names else 'First Names'
    
    print("=" * 60)
    print(f"Building {name_type_label} Filter List from Faker Package")
    print("=" * 60)
    
    # Determine which locales to use
    if args.locales:
        locales = args.locales
        print(f"Using specified locales: {', '.join(locales)}")
    elif args.discover:
        locales = discover_available_locales(name_type)
        print(f"Discovered {len(locales)} available locales")
    else:
        locales = None  # Use defaults
        print("Using default locale set")
    
    # Build the filter list
    all_names = build_name_filter_list(locales, name_type)
    
    print("\n" + "=" * 60)
    print(f"Total unique {name_type_label.lower()} collected: {len(all_names)}")
    print("=" * 60)
    
    if not all_names:
        print("ERROR: No names were collected. Please check:")
        print("  1. Faker package is installed correctly")
        print("  2. Locales are valid")
        print("  3. Person providers are available for the locales")
        sys.exit(1)
    
    # Save to file
    save_filter_list(all_names, args.output, name_type)
    
    # Print some statistics
    print("\nStatistics:")
    print(f"  Total names: {len(all_names)}")
    if all_names:
        sorted_all_names = sorted(list(all_names))
        max_display = args.max_display if args.max_display is not None else len(sorted_all_names)
        sample_names = sorted_all_names[:max_display]
        display_text = ', '.join(sample_names)
        if max_display < len(sorted_all_names):
            remaining = len(sorted_all_names) - max_display
            display_text += f" ... ({remaining} more not displayed)"
        print(f"  Sample names: {display_text}")
        # Show name length distribution
        name_lengths = [len(name) for name in all_names]
        if name_lengths:
            print(f"  Name length - Min: {min(name_lengths)}, Max: {max(name_lengths)}, Avg: {sum(name_lengths)/len(name_lengths):.1f}")


if __name__ == "__main__":
    main()
