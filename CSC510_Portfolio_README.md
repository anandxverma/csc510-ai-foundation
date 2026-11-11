# Neural Network-Based Decision Support for Trade Promotion Management

CSC510: Foundations of Artificial Intelligence - Portfolio Project (Anand Verma)

A decision-support program for a trade-marketing planner at a Clorox-sized consumer goods company.
The planner picks a promotion (category, retailer, weeks, discount, display, feature ad) and the program
reports predicted sales lift and profit, whether the promotion follows the business rules, and what the
best alternatives are. **The program never approves or rejects a promotion. The human planner decides.**

## Quick start

```
pip3 install -r CSC510_Portfolio_requirements.txt        # numpy and torch
python3 tpm_decision_support.py --demo       # complete walkthrough, nothing to type (about 15 seconds)
python3 tpm_decision_support.py              # interactive menu
python3 tpm_decision_support.py --selftest   # 18 automatic checks (about 15 seconds)
```

Options: `--seed N` (change the random seed; default 42) and `--quick` (smaller data, shorter training).
Tested with Python 3.12 on CPU. No internet connection and no outside data files are needed.

## What each menu option does

| Menu | Feature | Course concept |
|------|---------|----------------|
| 1 | Evaluate one promotion: predicted lift, incremental profit, ROI class, rule check, explanation | Deep learning, expert system, classification |
| 2 | Compare promotions side by side, route each one to the right reviewer, optionally save a review packet | Decision support, workflow layer |
| 3 | Find the best promotion for a category | Genetic algorithm (search) |
| 4 | Plan a multi-week promotion calendar | A* search, symbolic planning |
| 5 | Estimate a new product with no history | Nearest-neighbor search |
| 6 | Show model quality (test errors, ROI accuracy, cold-start test) | Evaluation |
| 7 | Show the business rules and planner operators | Knowledge representation |

Press Enter at any prompt to accept the value shown in brackets.

## How the pieces fit together

1. **Baseline model (not a neural network):** seasonal average of past non-promoted sales per category.
2. **Neural network (PyTorch):** embedding layers for category, display type, and month plus numeric
   inputs feed two hidden layers and predict % sales lift. Network settings are chosen by random search.
3. **Profit formula (not a neural network):** turns predicted lift into incremental profit and trade spend.
4. **Expert system:** a forward-chaining rule engine over first-order-style facts. Eleven rules with variables
   (retailer caps, shelf-group limits that products inherit, price floors, minimum margin after discount,
   spend limit, calendar overlap, and soft guidelines with certainty factors) label each promotion COMPLIANT,
   NEEDS REVIEW, or NON-COMPLIANT and print the chain of reasoning.
5. **Genetic algorithm:** searches 6,912 possible plans per category and retailer; fitness is the predicted
   profit minus penalties for rule problems. The demo checks it against scoring every plan.
6. **A* planner:** builds the best calendar from STRIPS-style operators (`Promote`, `DoNothing`) under a
   cooldown rule, repeat-promotion fatigue, a trade-spend budget, and a minimum number of promotions.
7. **Nearest neighbors:** a new product has no learned embedding yet, so it is described by five traits and
   borrows the network's predicted results for the five most similar categories.
8. **Workflow layer:** routes each result (non-compliant -> back to proposer, needs review -> trade marketing
   manager, allowed -> category planner) and can save `review_packet.txt`. In production this hand-off would
   move into a workflow platform such as Salesforce Agentforce.

## Important notes

* **Data are synthetic.** The Dunnhumby "Complete Journey" data cannot be bundled with this submission, so the
  program generates promotion history with similar fields (category, week, discount depth, display type,
  feature ad). Prices, margins, fees, caps, and the committed calendar are illustrative assumptions.
  Scores therefore show that the methods work on a known pattern, not how a real market behaves.
* Exact numbers can differ slightly between computers because of floating-point differences in PyTorch.
* The self-test compares A* with brute force on a small calendar, the genetic algorithm with exhaustive
  scoring, the rule engine with known cases, and the neural network with simple baselines.

## Files

* `tpm_decision_support.py` - the complete program (single file)
* `CSC510_Portfolio_requirements.txt` - libraries needed
* `CSC510_Portfolio_sample_output.txt` - output of `python3 tpm_decision_support.py --demo` (seed 42)
