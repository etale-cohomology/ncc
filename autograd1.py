'''
https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/supervised_learning/backpropagation
'''

# -----------------------------------------------------------------------------------------------------------------------------# @blk1
class OpAdd:
	def fwd(s, a,b):
		s.a = a
		s.b = b
		c   = a+b
		return c
	def bwd(s, dc):  # @arg dc  gradient of output c?
		da = 1*dc
		db = 1*dc
		return [da,db]  # return a gradient for each input...? WHAT???

class OpMul:
	def fwd(s, a,b):
		s.a = a
		s.b = b
		c   = a*b
		return c
	def bwd(s, dc):  # @arg dc  gradient of output c?
		da = s.b*dc
		db = s.a*dc
		return [da,db]  # return a gradient for each input...? WHAT???

class OpExp:
	pass

class OpLog:
	pass

# -----------------------------------------------------------------------------------------------------------------------------# @blk1
add = OpAdd()
mul = OpMul()

print(add.fwd(2,3))
print(mul.fwd(3,7))
print(add.bwd(11))
print(mul.bwd(11))
