'''
a system that learns the dice game Farkle.
Takeaways:
	1. large batch size is critical when there is high variance in outcomes; otherwise what a system absorbs is randomized by the scores between batches
	2. setting learning rate is hard; this is why ugly addition-based approaches work well.
	3. corollary: regularization sucks but is a reasonable way to keep weights in a reasonable size range
'''


import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

class Farkle:
	def __init__(self, verbosity=0):
		self.current_score = 0
		# self.theta = [20.0, -1, -.05, .05, 20]
		self.theta = np.random.rand(5)
		self.score_history = []
		self.v = verbosity
		
	def roll(self, n):
		# roll n dice
		dice = []
		for i in range(n):
			dice.append(random.randint(1,6))
		return dice
	
	def score(self, dice, greedy=0):
		# print(dice)
		# retrun score, number of leftover dice
		n = len(dice)
		if n > 6:
			print('invalid roll')
			return False
		c = Counter(dice)
		# 6 of a kind
		if max(c.values()) == 6:
			return 3000, 0
			
		# straight
		elif max(c.values()) == 1 and n == 6:
			return 1500, 0

		# 5 of a kind
		elif max(c.values()) == 5:
			if c[1] == 1:
				return 2100, 0
			elif c[5] == 1:
				return 2050, 0
			else:
				return 2000, n - 5

		# 4 of a kind, full house
		elif max(c.values()) == 4:
			# full house
			if min(c.values()) == 2:
				return 1500, 0
			if c[1] == 1 and c[5] == 1:
					return 1150, 0
				# less than 6 used
			if greedy:
				if c[1] == 1:
					return 1100, n - 5
				elif c[5] == 1:
					return 1050, n - 5
				else:
					return 1000, n - 4
			else:
				return 1500, n - 4
		
		# 3 of a kind
		elif max(c.values()) == 3:
			# double triple
			if list(c.values()) == [3, 3]:
				return 2500, 0
			# single three-of-a-kind
			else:
				# greedy
				if greedy or len(list(c.keys())) == 3:
					three_of = c.pop(c.most_common(1)[0][0])
					if three_of == 1:
						tempscore = 300
					else:
						tempscore = three_of * 100
					used = 3
					if c[1]:
						tempscore += c[1] * 100
						used += c[1]
					if c[5]:
						tempscore += c[5] * 50
						used += c[5]
					return tempscore, n - used
				else:
					'''
					# ultrapassive
					if c[1]:
						return 100, n - 1
					elif c[5]:
						return 50, n - 1

					else:
						three_of = c.pop(c.most_common(1)[0][0])
						return three_of * 100, n - 3
					'''
					three_of = c.pop(c.most_common(1)[0][0])
					if three_of == 1:
						tempscore = 300
					else:
						tempscore = three_of * 100
					return tempscore, n - 3

		# 3 pairs
		elif list(c.values()) == [2, 2, 2]:
			return 1500, 0

		# nothing
		else:
			if greedy or list(c.keys()) == [1, 5] or list(c.keys()) == [5, 1]:
				tempscore = c[1] * 100
				used = c[1]
				tempscore += c[5] * 50
				used += c[5]
				return tempscore, n - used
			else:
				if c[1]:
					return 100, n - 1
				elif c[5]:
					return 50, n - 1
				else:
					return 0, n
				
	def play(self):
		self.current_score = 0
		self._play(6)
				
	def _play(self, num_dice):
		dice = self.roll(num_dice)
		if self.v:
			print('rolled {}:'.format(num_dice), dice)
		tempscore, remaining_dice = self.score(dice)
		if tempscore:
			if self.v:
				print('scored', tempscore)
			self.current_score += tempscore

			if not remaining_dice:
				remaining_dice = 6

			decision = self.decide(remaining_dice)
			if decision:
					self._play(remaining_dice)
			else:
				if self.v:
					print('quit while ahead with {}'.format(self.current_score))
				self.score_history.append(self.current_score)
				return self.current_score
		
		# fail
		else:
			if self.v:
				if self.current_score:
					print('you lost your {} points by being greedy'.format(self.current_score))
				else:
					print('unlucky, score = 0')
			self.score_history.append(0)
			return 0
				
	def decide(self, num_dice):
		# return True to continue or False to stop.
		score = self.current_score
		t = self.theta
		activation = sum([num_dice * t[0],
											np.log10(score + .000001) * t[1],
											score * t[2],
											num_dice/(score + 1) * self.theta[3]])
		return activation > t[-1]
		
	def display(self):
		plt.figure()
		for dice in range(1, 7):
			for i in range(31):
				self.current_score = 100 * i
				choice = self.decide(dice)
				if choice:
					plt.scatter(dice, self.current_score, c='green')
				else:
					plt.scatter(dice, self.current_score, c='red')
		plt.show()
		plt.figure()
		plt.hist(self.score_history, bins=40)
		plt.show()


if __name__ == '__main__':
	f = Farkle()
	best_score = 0
	best_theta = f.theta.copy()
	
	games = 10000
	f.score_history = []
	
	# baseline
	for game in range(games):
		f.play()
	f.display()
	
	last = np.mean(f.score_history)
	best_time = 0
	ratio = 1e-4
		
	print(f.theta)

	# change parameters, if it works keep it
	for step in range(100):
		
		if step - best_time > 5:
			ratio *= 10
			best_time = step
			print('ratio =', ratio)
			if ratio > 10:
				f.theta = best_theta.copy()
				break
	
		for w in range(len(f.theta)):
			amt = (random.random() - .5) / ratio
			temp = f.theta[w]
			f.theta[w] *= 1 + amt

			f.score_history = []
			for _ in range(games):
				f.play()

			current = np.mean(f.score_history)	
			print('adjusting {} to {:.2}; best={}, current={}'.format(w, f.theta[w], best_score, current)) 
	
			if current >= last:
				last = current
			else:
				f.theta[w] = temp
				
			# print('best score =', best_score, 'current score =', current)
			
			if current > best_score:
				print('new best:', f.theta)
				best_time = step
				best_score = current
				best_theta = f.theta.copy()
		
	f.theta = best_theta.copy()
	
	print(f.theta)
	f.v = 1
	f.play()
	f.display()
