from __future__ import annotations
import requests
import urllib3
import re
import json
import types
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from owlready2 import get_ontology, Thing, AnnotationProperty
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@dataclass
class EventData:
    id: int
    title: str
    description: Optional[str] = None

class TextUtils:
    @staticmethod
    def clean_formatting(text: str) -> str:
        """
        Description:
        Cleans whitespace and sentence spacing format of a text string.

        Args:
            text (str): The raw text to clean.

        Returns:
            str: The cleaned text.
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(?<=[.\]])(?=[A-Z])', ' ', text)
        return text.strip()

    @staticmethod
    def format_class_name(text: str) -> str:
        """
        Description:
        Formats a string into a valid, safe class name by removing special characters and prefixing digits.

        Args:
            text (str): The raw string.

        Returns:
            str: The formatted class name string.
        """
        clean = text.strip().replace(" ", "_").replace("-", "_")
        clean = re.sub(r'[^\w_]', '', clean)

        if clean and clean[0].isdigit():
            clean = f"Event_{clean}"

        return clean

class AopScraper:
    BASE_URL = "https://aopwiki.org"

    def __init__(self) -> None:
        """
        Description:
        Initializes the AopScraper with requests session and retry strategy.

        Args:
            None

        Returns:
            None
        """
        self.session = requests.Session()
        self.session.verify = False

        # Add retry strategy for robustness
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def fetch_json_catalog(self) -> list[dict[str, Any]]:
        """
        Description:
        Fetches the complete catalog of key events from AopWiki in JSON format.

        Args:
            None

        Returns:
            list[dict[str, Any]]: A list of event catalogs.
        """
        try:
            print("Fetching event catalog JSON...")
            response = self.session.get(f"{self.BASE_URL}/events.json")
            response.raise_for_status()

            data = response.json()

            with open("events_catalog_raw.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            return data
        except Exception as e:
            print(f"Error fetching catalog: {e}")
            return []

    def fetch_event_description(self, event_id: int) -> str:
        """
        Description:
        Scrapes and extracts the key event description from the AopWiki event page.

        Args:
            event_id (int): The ID of the event to scrape.

        Returns:
            str: The cleaned description text.
        """
        url = f"{self.BASE_URL}/events/{event_id}"
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            header = soup.find(lambda t: t.name == "h2" and "Key Event Description" in t.get_text())
            if not header:
                return ""

            parent_section = header.find_parent("section")
            if parent_section:
                title_wrapper = parent_section.find("div", class_="section-h2")
                start_node = title_wrapper if title_wrapper else header
            else:
                start_node = header

            paragraphs = []
            for sibling in start_node.find_next_siblings():
                raw_text = sibling.get_text(" ", strip=True)
                cleaned = TextUtils.clean_formatting(raw_text)
                if cleaned:
                    paragraphs.append(cleaned)

            return "\n\n".join(paragraphs) if paragraphs else ""

        except Exception as e:
            print(f"Error scraping event {event_id}: {e}")
            return ""

class AopWorkflow:
    def __init__(self) -> None:
        """
        Description:
        Initializes the AopWorkflow pipeline.

        Args:
            None

        Returns:
            None
        """
        self.scraper = AopScraper()
        self.organized_events: Dict[str, List[EventData]] = defaultdict(list)

    def _print_category_preview(self, raw_events: list[dict[str, Any]]) -> None:
        """
        Description:
        Prints a statistics preview of events grouped by biological organization category.

        Args:
            raw_events (list[dict[str, Any]]): List of raw catalog events.

        Returns:
            None
        """
        print("\n--- Category Analysis ---")
        stats = Counter()
        for item in raw_events:
            org_info = item.get("biological_organization") or {}
            term = org_info.get("term", "Unknown")
            stats[term] += 1

        print(f"Found {len(stats)} distinct groups:")
        for term, count in stats.most_common():
            print(f"  - {term}: {count} events")
        print("-------------------------\n")

    def run(self) -> None:
        """
        Description:
        Runs the full scraping workflow, fetching catalogs and scraping event pages.

        Args:
            None

        Returns:
            None
        """
        raw_events = self.scraper.fetch_json_catalog()
        if not raw_events: return

        self._print_category_preview(raw_events)

        total_events = len(raw_events)
        print(f"Starting detailed processing of {total_events} events...")

        for i, item in enumerate(raw_events, 1):
            org_info = item.get("biological_organization") or {}
            term = org_info.get("term", "Unknown")
            event_id = item.get("id")
            title = item.get("title")

            print(f"[{i}/{total_events}] Processing ID {event_id} [{term}]...")
            description = self.scraper.fetch_event_description(event_id)

            event_obj = EventData(id=event_id, title=title, description=description)
            self.organized_events[term].append(event_obj)

    def load_from_json(self, filename: str = "events_with_descriptions.json") -> None:
        """
        Description:
        Loads scraped events and their descriptions from a local JSON file.

        Args:
            filename (str): The JSON catalog filename. Defaults to "events_with_descriptions.json".

        Returns:
            None
        """
        print(f"Loading data from '{filename}'...")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            count = 0
            for term, events_list in raw_data.items():
                for event_dict in events_list:
                    event_obj = EventData(**event_dict)
                    self.organized_events[term].append(event_obj)
                    count += 1

            print(f"Successfully loaded {count} events across {len(self.organized_events)} categories.")

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found. Please check the path.")
        except Exception as e:
            print(f"Error loading JSON: {e}")

    def save_to_json(self, filename: str = "events_with_descriptions.json") -> None:
        """
        Description:
        Saves structured events and descriptions to a local JSON file.

        Args:
            filename (str): The target JSON filename. Defaults to "events_with_descriptions.json".

        Returns:
            None
        """
        print(f"\nSaving JSON to '{filename}'...")
        output_data = {k: [asdict(e) for e in v] for k, v in self.organized_events.items()}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print("JSON Saved.")

    def save_to_owl(self, filename: str = "aop_events_hierarchy.owl") -> None:
        """
        Description:
        Builds and saves the owlready2 ontology for the AOP events hierarchy.

        Args:
            filename (str): The output OWL filename. Defaults to "aop_events_hierarchy.owl".

        Returns:
            None
        """
        if not self.organized_events:
            print("No data to convert to OWL.")
            return

        print(f"\nGenerating Ontology '{filename}'...")

        onto = get_ontology("http://aopwiki.org/ontology#")

        with onto:
            class has_description(AnnotationProperty):
                pass

            class has_aop_id(AnnotationProperty):
                pass

            for group_term, events in self.organized_events.items():
                if group_term == "Unknown":
                    print(f"  > Skipping excluded group: {group_term} ({len(events)} events removed)")
                    continue

                parent_name = TextUtils.format_class_name(group_term)
                if not parent_name:
                    continue

                ParentClass = types.new_class(parent_name, (Thing,))

                print(f"  > Processing Group: {parent_name}")

                for event in events:
                    safe_title = TextUtils.format_class_name(event.title)
                    class_name = f"{safe_title}_{event.id}"

                    EventClass = types.new_class(class_name, (ParentClass,))

                    EventClass.label.append(event.title)
                    EventClass.has_aop_id.append(event.id)

                    if event.description:
                        clean_desc = event.description.replace("\x00", "")
                        EventClass.has_description.append(clean_desc)

        onto.save(file=filename, format="rdfxml")
        print(f"Done! Ontology saved to {filename}")

if __name__ == "__main__":
    workflow = AopWorkflow()

    workflow.run()
    workflow.save_to_json("events_with_descriptions.json")

    workflow.load_from_json("events_with_descriptions.json")
    workflow.save_to_owl("aop_events_hierarchy.owl")
