# CAC 40 Stock Analysis — French Market Deep Dive

## Overview
Python-based quantitative analysis of 15 major CAC 40 companies across 7 sectors,
covering returns, volatility, correlation, Sharpe ratios, and sector performance.

## How to Run on Your Mac
```bash
pip3 install yfinance pandas matplotlib seaborn
python3 cac40_analysis.py
```

## Charts Generated
1. **Risk-Return Scatter** — The classic portfolio analysis chart (risk vs return by sector)
2. **Correlation Heatmap** — Which stocks move together (diversification analysis)
3. **Cumulative Returns** — €1 invested growth visualization
4. **Sector Performance** — Average return & volatility by sector
5. **Sharpe Ratio Ranking** — Risk-adjusted performance (the #1 metric)
6. **Rolling Volatility** — How risk changes over time (30-day window)

## Stocks Analyzed (15 companies, 7 sectors)
| Company | Sector | Ticker |
|---------|--------|--------|
| LVMH | Luxury | MC.PA |
| Hermès | Luxury | RMS.PA |
| L'Oréal | Consumer | OR.PA |
| Kering | Luxury | KER.PA |
| Airbus | Aerospace | AIR.PA |
| Safran | Aerospace | SAF.PA |
| Schneider Electric | Industrials | SU.PA |
| BNP Paribas | Banking | BNP.PA |
| Crédit Agricole | Banking | ACA.PA |
| Sanofi | Healthcare | SAN.PA |
| Air Liquide | Industrials | AI.PA |
| TotalEnergies | Energy | TTE.PA |
| Vinci | Industrials | DG.PA |
| AXA | Insurance | CS.PA |
| Dassault Systèmes | Technology | DSY.PA |

## Key Concepts Explained (in the code comments)
- **Daily Returns** — Why we use returns instead of prices
- **Annualized Volatility** — How to scale daily risk to annual
- **Sharpe Ratio** — Risk-adjusted return formula
- **Correlation** — Why it matters for diversification
- **Rolling Volatility** — Real-time risk monitoring

## Skills Demonstrated
Python, pandas, matplotlib, seaborn, Financial Statistics, French Market Knowledge

## Author
Akshat Sharma — MSc Financial Analysis, INSEEC Paris — March 2026
