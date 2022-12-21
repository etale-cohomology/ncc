'''
http://utkuevci.com/ml/autograd/
'''
class Variable():
	__counter = 0

	# Each Variable need its data which is a scalar or a numpy.ndarray if it is not a leaf node we need the backward_fun. __counter is an internal counter for debugging purposes. self.prev is an array pointing the parents and initialized as an empty array: should be set manually after creation. Backward function is called on the self.grad so we should guarantee that it is fully accumulated before calling the backward on the Variable.
	def __init__(self,data,is_leaf=True,backward_fun=None):
		if backward_fun is None and not is_leaf:  raise ValueError('non leaf nodes require backward_fun')
		if np.isscalar(data):  data = np.ones(1)*data
		if not isinstance(data,np.ndarray):  raise ValueError(f'data should be of type "numpy.ndarray" or a scalar,but received {type(data)}')
		self.data          = data
		self.id            = Variable.__counter
		Variable.__counter += 1
		self.is_leaf       = is_leaf
		self.prev          = []
		self.backward_fun  = backward_fun
		self.zero_grad()

	def backward(self):   self.backward_fun(dy=self.grad)
	def zero_grad(self):  self.grad = np.zeros(self.data.shape)
	def step(self,lr):    self.data -= lr*self.grad
	def __str__(self):    return f'Variable(id:{self.id},prev:{list(map(lambda a:a.id,self.prev))},is_leaf:{self.is_leaf})\n'

# Each operation creates the backward_fun of the new Variable as a closure bound the the the parents. One can implement this part with generic functions which take the parents each time as parameters. This is possible and might lead to a more efficient run-time performance. However, this is not our primary concern here, so we go with the closures. backward_fun of the dot is simple, just the dot product of the dy with the other Variable’s data.
def plus(a,b):
	if not (isinstance(a,Variable) and isinstance(b,Variable)):
		raise ValueError('a,b needs to be a Variable instance')
	def b_fun(dy):
		b.grad += dy
		a.grad += dy

	res = Variable(a.data+b.data,is_leaf=False,backward_fun=b_fun)
	res.prev.extend([a,b])
	return res

def plus_bcast(a,b):
	# a being a matrix(mini-batch output m*n)
	# b being a vector(bias n)
	if not (isinstance(a,Variable) and isinstance(b,Variable)):
		raise ValueError('a,b needs to be a Variable instance')
	def b_fun(dy):
		b.grad += dy.sum(axis=0)
		a.grad += dy

	res = Variable(a.data+b.data,is_leaf=False,backward_fun=b_fun)
	res.prev.extend([a,b])
	return res

def minus(a,b):
	if not (isinstance(a,Variable) and isinstance(b,Variable)):  raise ValueError('a,b needs to be a Variable instance')
	def b_fun(dy):
		b.grad += -dy
		a.grad += dy
	res = Variable(a.data-b.data,is_leaf=False,backward_fun=b_fun)
	res.prev.extend([a,b])
	return res

def multiply(a,b):
	if not (isinstance(a,Variable) and isinstance(b,Variable)):  raise ValueError('a,b needs to be a Variable instance')
	def b_fun(dy):
		if np.isscalar(dy):
			dy = np.ones(1)*dy
		a.grad += np.multiply(dy,b.data)
		b.grad += np.multiply(dy,a.data)
	res = Variable(np.multiply(a.data,b.data),is_leaf=False,backward_fun=b_fun)
	res.prev.extend([a,b])
	return res

def c_mul(a,c):
	if not (isinstance(a,Variable) and isinstance(c,(int, float))):
		raise ValueError('a needs to be a Variable, c needs to be one of (int, float)')
	def b_fun(dy=1):
		a.grad += dy*c
	res = Variable(a.data*c,is_leaf=False,backward_fun=b_fun)
	res.prev.append(a)
	return res

def dot(a,b):
	if not (isinstance(a,Variable) and isinstance(b,Variable)):  raise ValueError('a,b needs to be a Variable instance')
	def b_fun(dy):
		if np.isscalar(dy):
			dy = np.ones(1)*dy
		a.grad += np.dot(dy,b.data.T)
		b.grad += np.dot(a.data.T,dy)
	res = Variable(np.dot(a.data,b.data),is_leaf=False,backward_fun=b_fun)
	res.prev.extend([a,b])
	return res

def sumel(a):  # sumel is just a broadcast when we look at the backward pass.
	if not (isinstance(a,Variable)):  raise ValueError('a needs to be a Variable')
	def b_fun(dy=1):
		a.grad += np.ones(a.data.shape)*dy

	res = Variable(np.sum(a.data),is_leaf=False,backward_fun=b_fun)
	res.prev.append(a)
	return res

def relu(a):  # backward_fun of the relu is just the masking.
	if not (isinstance(a,Variable)):  raise ValueError('a needs to be a Variable')
	def b_fun(dy=1):  a.grad[a.data>0] += dy[a.data>0]

	res = Variable(np.maximum(a.data, 0),is_leaf=False,backward_fun=b_fun)
	res.prev.append(a)
	return res

def transpose(a):
	if not (isinstance(a,Variable)):  raise ValueError('a needs to be a Variable')
	def b_fun(dy=1):
		a.grad += dy.T

	res = Variable(a.data.T,is_leaf=False,backward_fun=b_fun)
	res.prev.append(a)
	return res

# What we need to do is to call .backward() on each variable that is in our computational graph. We have the whole graph for every Variable since each Variable points its parents. The trick here is the call the .backward() in the right order since we need the .grad of the Variable to be fully accumulated before its .backward() call. To ensure this we do a topological sort and call the .backward() accordingly.
def backward_graph(var):
	if not isinstance(var,Variable):
		raise ValueError('var needs to be a Variable instance')
	tsorted = __top_sort(var)

	var.grad=np.ones(var.data.shape)
	for var in reversed(tsorted):
		var.backward()

def __top_sort(var):
	vars_seen = set()
	top_sort = []
	def top_sort_helper(vr):
		if (vr in vars_seen) or vr.is_leaf:
			pass
		else:
			vars_seen.add(vr)
			for pvar in vr.prev:  top_sort_helper(pvar)
			top_sort.append(vr)
	top_sort_helper(var)
	return top_sort

class LinearLayer:
	def __init__(self,features_inp,features_out):
		super(LinearLayer, self).__init__()
		std = 1.0/features_inp
		self.w = Variable(np.random.uniform(-std,std,(features_inp,features_out)))
		self.b = Variable(np.random.uniform(-std,std,features_out))

	def forward(self, inp):
		return plus_bcast(dot(inp,self.w),self.b)

	def zero_grad(self):
		self.w.zero_grad()
		self.b.zero_grad()

	def step(self,lr):
		self.w.step(lr)
		self.b.step(lr)
