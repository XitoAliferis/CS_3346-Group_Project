from __future__ import annotations #i hate python.
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ortools.sat.python import cp_model

@dataclass
class NQueensLayout:
	n: int
	queenColumnPositions: List[int]

	def isValid(self) -> bool:
		for i in range(self.n):
			for j in range(i+1, self.n):
				if self.queenColumnPositions[i] == self.queenColumnPositions[j] or abs(self.queenColumnPositions[i] - self.queenColumnPositions[j]) == j - i:
					return False
		return True
	
	def toTextTable(self) -> str:
		table = []
		for i in range(self.n):
			row = []
			for q in range(self.n):
				if self.queenColumnPositions[q] == i:
					row.append("X")
				else:
					row.append("∙")
			table.append(row)
		return "\n".join([" ".join(row) for row in table])
	
	def toTextColumnList(self) -> str:
		return ",".join([str(i) for i in self.queenColumnPositions])
	
	def asDict(self) -> Dict[str, Any]:
		return {
			"n": self.n,
			"queenColumnPositions": self.queenColumnPositions,
		}

	
	@staticmethod
	def fromPositions(positions: List[int]) -> NQueensLayout:
		return NQueensLayout(len(positions), positions)


def calculateNQueensSolutions(n: int, maxCount: int, logProgress: bool = False) -> List[NQueensLayout]: #Following https://developers.google.com/optimization/cp/queens
	model = cp_model.CpModel()
	queenVars = [model.new_int_var(0, n - 1, f"x_{i}") for i in range(n)]

	model.add_all_different(queenVars)
	model.add_all_different(queenVars[i] + i for i in range(n))
	model.add_all_different(queenVars[i] - i for i in range(n))

	class NQueenSolutionPrinter(cp_model.CpSolverSolutionCallback):
		def __init__(self, queens: list[cp_model.IntVar], maxCount: int, outputList: List[NQueensLayout], logProgress: bool = False):
			cp_model.CpSolverSolutionCallback.__init__(self)
			self.__solved = 0
			self.__queens = queens
			self.__maxCount = maxCount
			self.__outputList = outputList

		def on_solution_callback(self):
			self.__outputList.append(NQueensLayout.fromPositions([self.Value(var) for var in self.__queens]))
			self.__solved += 1
			if logProgress and (self.__solved % 100 == 0 or self.__maxCount <= 100):
				print(f"[NQueensSolver] Solving for n = {n}, found {self.__solved} of {self.__maxCount} requested")
			if self.__solved >= self.__maxCount:
				self.StopSearch()
	
	results = []
	solver = cp_model.CpSolver()
	solutionPrinter = NQueenSolutionPrinter(queenVars, maxCount, results)
	solver.parameters.enumerate_all_solutions = True
	if logProgress:
		print(f"[NQueensSolver] Starting solve for n = {n}")
	solver.solve(model, solutionPrinter)
	if logProgress:
		print(f"[NQueensSolver] Done for n = {n}, found {len(results)} of {maxCount} requested ({len(results) / maxCount * 100:.2f}%)")
	return results

def calculateCumulativeNQueensSolutions(amount: int, minimumN: int = 4, logProgress: bool = False) -> List[NQueensLayout]:
	n = minimumN
	solutions = []
	while len(solutions) < amount:
		solutions.extend(calculateNQueensSolutions(n, amount - len(solutions), logProgress))
		n += 1
	return solutions


def calculateBucketedCumulativeNQueensSolutions(amount: int, minimumN: int = 4, logProgress: bool = False) -> Dict[int, List[NQueensLayout]]:
	n = minimumN
	solved = 0
	solutions = {}
	while solved < amount:
		solutions[n] = calculateNQueensSolutions(n, amount - solved, logProgress)
		solved += len(solutions[n])
		n += 1
	return solutions

def getFirstAmountOfBuckets(amount: int, bucketsIn: Dict[int, List[NQueensLayout]]) -> Dict[int, List[NQueensLayout]]:
	bucketsOut = {}
	totalMoved = 0
	for n, solutions in bucketsIn.items():
		if len(solutions) > amount - totalMoved:
			bucketsOut[n] = solutions[:amount - totalMoved]
			totalMoved += len(bucketsIn[n])
		else: 
			bucketsOut[n] = solutions
			totalMoved += len(solutions)
		if totalMoved >= amount:
			break
	return bucketsOut


def splitBucketedSolutionsByPercentage(percent: float, buckets: Dict[int, List[NQueensLayout]], randomSeed: int = 0) -> Tuple[List[NQueensLayout], List[NQueensLayout]]:
	percentSelected = []
	rest = []
	rng = random.Random(randomSeed)
	for n, solutions in buckets.items():
		shuffled = solutions.copy()
		rng.shuffle(shuffled)
		percentSelected.extend(shuffled[:int(len(shuffled) * percent)])
		rest.extend(shuffled[int(len(shuffled) * percent):])
	return percentSelected, rest

def generate_nqueens_dataset(
		amountForTesting: int, 
		firstSplit: int, 
		secondSplit: int, 
		thirdSplit: int, 
		minimumN: int = 4,
		logProgress: bool = False, 
		randomSeed: int = 0
	) -> Tuple[
		List[Dict[str, Any]], # firstSplitTrain
		List[Dict[str, Any]], # firstSplitTest
		List[Dict[str, Any]], # secondSplitTrain
		List[Dict[str, Any]], # secondSplitTest
		List[Dict[str, Any]], # thirdSplitTrain
		List[Dict[str, Any]]  # thirdSplitTest
	]:
	# (we assume the 3rd split is the biggest)
	thirdSplitBuckets = calculateBucketedCumulativeNQueensSolutions(thirdSplit, minimumN=minimumN, logProgress=logProgress)
	firstSplitBuckets = getFirstAmountOfBuckets(firstSplit, thirdSplitBuckets)
	secondSplitBuckets = getFirstAmountOfBuckets(secondSplit, thirdSplitBuckets)

	firstSplitTest, firstSplitTrain = splitBucketedSolutionsByPercentage(amountForTesting/firstSplit, firstSplitBuckets, randomSeed=randomSeed)
	secondSplitTest, secondSplitTrain = splitBucketedSolutionsByPercentage(amountForTesting/secondSplit, secondSplitBuckets, randomSeed=randomSeed)
	thirdSplitTest, thirdSplitTrain = splitBucketedSolutionsByPercentage(amountForTesting/thirdSplit, thirdSplitBuckets, randomSeed=randomSeed)

	return (
		[solution.asDict() for solution in firstSplitTrain],
		[solution.asDict() for solution in firstSplitTest],
		[solution.asDict() for solution in secondSplitTrain],
		[solution.asDict() for solution in secondSplitTest],
		[solution.asDict() for solution in thirdSplitTrain],
		[solution.asDict() for solution in thirdSplitTest]
	)
