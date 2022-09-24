

def forward_diff(func, a, h):
    """
    Numerical differentiation scheme which gets the derivative of func at a 
    by computing the value of the function at some a + h > a
    """
    return (func(a + h) - func(a)) / h

def central_diff(func, a, h):
    """
    Numerical differentiation scheme which gets the derivative of func at a 
    by computing the value of the function at some a + h / 2 > a
    and some a - h / 2 < a
    """
    return (func(a + h / 2) - func(a - h / 2)) / h