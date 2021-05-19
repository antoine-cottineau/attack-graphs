import json
import numpy as np
import random
import requests
import utils
from pathlib import Path
from typing import List


class ExploitFetcher:
    def __init__(self):
        self.fake_exploits_file = Path(
            "methods_input/exploits/fake_exploits.json")

    def get_fake_exploit_list(self,
                              n_exploits: int,
                              update_file: bool = False) -> List[dict]:
        if self.fake_exploits_file.exists() and not update_file:
            # Load the existing file of fake exploits
            with open(self.fake_exploits_file, "r") as f:
                all_exploits = json.load(f)

            # Sample n_exploits exploits from this list
            ids_exploits = np.random.choice(len(all_exploits), size=n_exploits)
            exploits = [all_exploits[i] for i in ids_exploits]
            return exploits
        else:
            # Generate a fake list of exploits
            exploits = self._generate_exploit_list(n_exploits)

            # Save the list in the file
            utils.create_parent_folders(self.fake_exploits_file)
            with open(self.fake_exploits_file, "w") as f:
                json.dump(exploits, f, indent=2)

            return exploits

    def _generate_exploit_list(self, n_exploits: int) -> List[dict]:
        # Generate a fake list of CVE ids
        cve_ids = [
            "cve-2020-{:04.0f}".format(random.randint(1, 9999))
            for _ in range(n_exploits)
        ]

        # Fetch each one of the fake exploits
        exploits = []
        for cve_id in cve_ids:
            exploit = self._get_exploit_from_cve_id(cve_id)
            if exploit is not None:
                exploits.append(exploit)

        # Some of the exploits may not exist. Thus, we must fill the list of
        # exploits with new exploits
        while len(exploits) < n_exploits:
            exploits += self._generate_exploit_list(n_exploits - len(exploits))

        return exploits

    def _get_exploit_from_cve_id(self, cve_id) -> dict:
        # Create the url to fetch
        url = "http://api.cvesearch.com/search?q={}".format(cve_id)

        # Fetch the url
        response = requests.request("GET", url)

        # Parse it
        json_object = json.loads(response.text)

        # Get the description text
        text = json_object["response"][cve_id]["basic"]["description"]

        # If the text is N/A, the cve_id does not correspond to an existing
        # exploit
        if text == "N/A":
            return None

        # Get the CVSS score
        cvss = float(
            json_object["response"][cve_id]["details"]["cvssV3_score"])

        # Some exploits are wrongly entered in the database. Thus, we check
        # that the cvss score is superior to 0
        if cvss == 0:
            return None

        # Return the exploit dictionary
        exploit = dict(cve_id=cve_id, text=text, cvss=cvss)
        return exploit
