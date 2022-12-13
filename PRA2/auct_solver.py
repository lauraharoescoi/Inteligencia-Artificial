import argparse
import collections
import itertools 
import os
import sys

import msat_runner
import wcnf

class Bid(object):
    def __init__(self, line=""):
        self.agent = line[0]
        self.goods = [g for g in line[1:-1]]
        self.cost = int(line[-1])

    def toString(self):
        string = ""
        string += "agent: " + self.agent + "\n"
        string += "goods: "
        for g in self.goods:
            string += g+" " 
        string += "\ncost: " + str(self.cost) + "\n"
        return string

class AuctionProblem(object):
    """This class represents a combinatorial auction problem. The agents are
    labeled 1, ..., k, where k is the number of agents, the goods are labeled
    1, ..., n, where n is the number of goods, and the bids are stored as pairs of 
    1 agent and 1 good.
    """

    def __init__(self, file_path=""):
        self.agents = []
        self.n_agents = 0
        self.goods = []
        self.n_goods = 0
        self.bids = []

        if file_path:
            self.read_file(file_path)

    def read_file(self, file_path):
        """Loads an auction from the given file.

        :param file_path: Path to the file that contains an auction definition.
        """
        with open(file_path, 'r') as stream:
            self.read_stream(stream)

    def read_stream(self, stream):
        """Loads an auction from the given stream.

        :param stream: A data stream from which read the auction problem definition.
        """
        n_agents = -1
        agents = []
        n_goods = -1
        goods = []

        reader = (l for l in (ll.strip() for ll in stream) if l)
        for line in reader:
            l = line.split()
            if l[0] == 'a':
                self.n_agents = len(l) - 1
                for agent in l[1:]:
                    self.agents.append(agent)
            elif l[0] == 'g':
                self.n_goods = len(l) - 1
                for good in l[1:]:
                    self.goods.append(good)
            elif l[0] == 'c':
                pass  # Ignore comments
            else:
                self.bids.append(Bid(l))

    def MaxSatSolve(self, solver, no_min_win):
        formula = wcnf.WCNFFormula()

        nodes = [formula.new_var() for _ in self.bids]
        for i, bid in enumerate(self.bids):
            formula.add_clause([nodes[i]], bid.cost)

        for b1 in range(len(self.bids) -1):
            for b2 in range(b1 + 1, len(self.bids)):
                intersection = [value for value in self.bids[b1].goods if value in self.bids[b2].goods]
                if intersection != []:
                    formula.add_clause([-nodes[b1], -nodes[b2]])
        
        if no_min_win:
            _, model = solver.solve(formula)
            return [n for n in model if n > 0]
        
        for agent in self.agents:
            bids = []
            for bid in range(len(self.bids)):
                if agent == self.bids[bid].agent:
                    bids.append(nodes[bid])
            formula.add_clause(bids)

        _, model = solver.solve(formula)
        return [n for n in model if n > 0]

def main(argv=None):
    args = parse_command_line_arguments(argv)
    solver = msat_runner.MaxSATRunner(args.solver)
    auction = AuctionProblem(args.auction)
    auction_solved = auction.MaxSatSolve(solver, args.no_min_win_bids)
    print("APS", " ".join(map(str, auction_solved)))

def parse_command_line_arguments(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("solver", help="Path to the MaxSAT solver.")

    parser.add_argument("auction", help="Path to the file that descrives the"
                                      " input combinatorial auctions problem.")
    
    parser.add_argument("--no-min-win-bids", action="store_true",
                        help="Deactivate the minimum winning bids constraints")

    return parser.parse_args(args=argv)



if __name__ == "__main__":
    sys.exit(main())
