# Hands-on page review

September 19, 2026. Tested the running page in the in-app browser, including screenshots at the user's narrow panel size.

## Reproduced and corrected

- Category selection changed only a local filter, leaving other categories empty. It now requests live markets automatically.
- Discovery stopped after three series even when those series had no open markets. It now searches up to 30 non-weather series, stopping after three series with usable markets. The board displays how much of the catalog was checked; it does not imply complete exchange coverage. The live Mentions check found 14 contracts.
- The old Browse all markets action did not clear search or saved/model filters. Renamed it Reset filters & browse and made it clear them.
- A loading notice remained after successful refresh. Requests now track their job to success or failure and show a completed status.
- An analyzed link could be hidden by filters from a previous search. Successful analysis resets the filters and switches the board only when results are ready.
- Refreshing an analyzed link switched away to category discovery. It now refreshes the current analyzed reference.
- Old card recommendations could remain visible after quotes aged. Cards now evaluate quote age on each render.
- On narrow screens the wallet was below the market list. Added a Practice setup shortcut that scrolls to the wallet and focuses the balance.
- Selecting a contract for practice was hidden in advanced controls. A visible selection summary and clear button now expose it; changing scope clears the override.
- Practice initially selected an old analyzed link. New page loads default to automatic daily-weather discovery.
- Mentions contracts shared a generic event title. Their specific word/phrase is now prominent on the card and detail panel.
- A reload could restore a cached board with the wrong category selected. The initial category now matches the cached board.

## Verification

- Live category fetch found 14 mentions contracts after checking 30 of 451 series.
- Entering a deliberately unmatched search hid the cards; Reset filters & browse restored all 14.
- A browsed mention contract opened fresh details and could be selected for practice; the exact ticker appeared in the wallet and the clear button removed it.
- The practice shortcut visibly landed at the wallet with the balance focused.
- JavaScript syntax checks passed, with no console errors in the tested flows.
- All **90 Python tests passed**, including a new regression proving discovery continues past inactive series.

No real-money transactions were made. These improvements change usability and discovery, not the evidence for model profitability; the limitations in AUDIT.md remain.
