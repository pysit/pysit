#include <iostream>

/*

This file is a demonstration of how the Actor/Action/ActionArgument pattern, used for unrolling if statements in templates, works.

It is far easier to follow than the actual if unroller code, and should function as a good "getting started" point.

*/

template <typename T>
class ActionArgument
{
    public:
        T & a;

        ActionArgument(T &aa):a(aa){};
};

template <typename T,
          template <typename A> class ACTIONARG>
class Action
{
    public:

        static
        void process(ACTIONARG<T>& arg)
        {
            arg.a += 5;
            std::cout << arg.a << std::endl;
        };
};

template <typename T,
          template <typename A, template <typename D> class B> class ACTION,
          template <typename C> class ACTIONARG>
class Actor
{
    public:
        static
        void exec(ACTIONARG<T>& arg)
        {
            ACTION < T, ACTIONARG >::process(arg);
        };
};


int main(int argc, char** argv)
{
    int q = 4;
    ActionArgument<int> a(q);

    Action<int, ActionArgument>::process(a);
    std::cout << q << std::endl;

    Actor<int, Action, ActionArgument>::exec(a);
    std::cout << q << std::endl;
}
