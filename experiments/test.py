class A(object):
    def foo(self, foo1):
        a = 42 + foo1
        print(a)
        return a


class B(A):
    def foo(self):
        foo1 = 5
        b = super(B, self).foo(foo1)
        b += 1
        print(b)
        return b

def main():
    b = B()
    b.foo()
main()
