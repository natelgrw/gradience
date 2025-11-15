#!/usr/bin/env python3
"""
askcos_scraper.py

Author: natelgrw
Last Edited: 11/15/2025

This script contains a function that scrapes the ASKCOS forward prediction
results for a given list of reactants and solvent.
"""

import asyncio
import os
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors
from playwright.async_api import async_playwright


# ===== Scraper Function ===== #


async def scrape_askcos(reactant_smiles_list, solvent_smiles):
    """
    Scrape ASKCOS forward prediction results for given reactants and solvent.
    
    Args:
        reactant_smiles_list (list): List of reactant SMILES strings
        solvent_smiles (str): Solvent SMILES string
        output_filename (str): Name of the output CSV file (default: "forward.csv")
    
    Returns:
        str: Path to the downloaded CSV file
    """
    # join reactant SMILES with periods
    combined_reactants = ".".join(reactant_smiles_list)
    print(f"Combined reactants: {combined_reactants}")

    async with async_playwright() as p:
        # launch Chromium headless browser
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # navigate to ASKCOS forward prediction page
        print("Navigating to ASKCOS forward page...")
        await page.goto("https://askcos.mit.edu/forward", wait_until="networkidle")

        # wait for the button by text content using a locator
        print("Navigating to Product Prediction tab...")
        product_button = page.locator("button", has_text="Product Prediction")
        await product_button.wait_for(timeout=20000)
        await product_button.click()
        print("Clicked Product Prediction tab")

        # fill reactants
        print("Navigating to Reactants tab")
        reactant_input = page.locator("input[placeholder='SMILES'][id^='input-']")
        await reactant_input.first.fill(combined_reactants)
        print("Entered Reactants")

        # fill solvents
        print("Navigating to Solvents tab")
        reactant_input = page.locator("input[placeholder='SMILES'][id^='input-']")
        await reactant_input.nth(2).fill(solvent_smiles)
        print("Entered Solvents")

        # get results
        print("Navigating to Results button...")
        get_results_button = page.locator("button:has-text('Get Results')")
        await get_results_button.click()
        print("Clicked Get Results button")

        await asyncio.sleep(10)

        # export and save prediction file
        print("Navigating to Export button...")
        get_results_button = page.locator("button:has-text('Export')")
        await get_results_button.click()
        print("Clicked Export button")

        download_path = os.path.dirname(os.path.abspath(__file__))

        async with page.expect_download() as download_info:
            get_results_button = page.locator("button:has-text('Save')")
            await get_results_button.click()
        download = await download_info.value

        dest_file = os.path.join(download_path, "forward.csv")
        await download.save_as(dest_file)

        await asyncio.sleep(5)

        results = []

        for smiles in reactant_smiles_list:
            mol_reactant = Chem.MolFromSmiles(smiles)
            mw_reactant = Descriptors.ExactMolWt(mol_reactant)
            results.append({
                "smiles": smiles,
                "probability": 1,
                "mol_weight": mw_reactant,
            })

        mol_solvent = Chem.MolFromSmiles(solvent_smiles)
        mw_solvent = Descriptors.ExactMolWt(mol_solvent)
        results.append({
                "smiles": solvent_smiles,
                "probability": 1,
                "mol_weight": mw_solvent,
            })

        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(script_dir, "forward.csv")
        with open(csv_file, "r") as file:
            reader = csv.reader(file)
            next(reader)  # skip header row
            for row in reader:
                results.append({
                    "smiles": row[1],
                    "probability": row[2],
                    "mol_weight": row[4],
                })

        if os.path.exists(csv_file):
            os.remove(csv_file)
        else:
            print(f"File not found: {csv_file}")

        return results


# ===== Main ===== #


async def main():
    """
    Example usage of the scrape_askcos function.
    """
    reactants = ["CC(=O)OC(C)=O"]
    solvent = "CCO"
    
    results = await scrape_askcos(reactants, solvent)
    print(results)


if __name__ == "__main__":
    asyncio.run(main())