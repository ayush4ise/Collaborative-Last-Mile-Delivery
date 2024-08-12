README

Folder Contents:
- This folder contains two subfolders, one for smaller instances and one for larger instances.
- The smaller instances are solved by three different approaches as described in the paper, each represented in a separate subfolder for each instance.
- The naming convention for smaller instance results is as follows:
  - 'total_cost_i': 
    - 'i' represents the approach used:
      - 1: Non-collaborative case
      - 2: Collaborative case with sharing assets
      - 3: Proposed collaborative case without sharing assets
	- All the above three are solved by an exact method using CPLEX.
      - 4: Proposed approach of collaborating without asset sharing with proposed heuristics
    
- The larger instances are organized straightforwardly, with each logistics service provider (LSP) having independent distribution without collaboration (exact method) and then the proposed approach presented and solved using heuristics.

Usage:
- Navigate to the desired subfolder based on the instance size and approach used.
- For smaller instances, select the appropriate instance folder to access the corresponding results.
- For larger instances, select the relevant instance folder to view the results for independent distribution and the proposed collaborative approach.
