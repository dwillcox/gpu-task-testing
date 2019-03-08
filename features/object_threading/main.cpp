#include <iostream>
#include <thread>

class A {

    bool is_active;

    std::thread* dedicated_thread;

public:

    A(std::function<void()> thread_fun) {
        std::cout << "constructing object now with function argument" << std::endl;
        construct_thread(thread_fun);
    }

    A() {
        std::cout << "constructing object now with no argument" << std::endl;
        construct_thread([&]{thread_operation();});
    }

    ~A() {
        is_active = false;
        dedicated_thread->join();
        delete dedicated_thread;
    }

    void construct_thread(std::function<void()> thread_fun) {
        is_active = true;
        dedicated_thread = new std::thread([&, thread_fun]{thread_polling(thread_fun);});
    }

    void thread_polling(std::function<void()> fun_to_poll) {
        while (is_active) {
            fun_to_poll();
        }
    }

    void thread_operation() {
        std::cout << "thread in class A is polling" << std::endl;
    }
    
};


class B {

    A* aptr;

public:

    B() {
        aptr = new A([&]{thread_operation();});
    }

    ~B() {
        delete aptr;
    }
    
    void thread_operation() {
        std::cout << "thread in class B is polling" << std::endl;
    }
};


class C : public A {
public:

    C() : A([&]{thread_operation();}) {}
    
    void thread_operation() {
        std::cout << "thread in class C is polling" << std::endl;
    }
};


int main(int argc, char* argv[]) {

    std::cout << "Starting program" << std::endl;

    {
        A a;

        std::cout << "Created object a" << std::endl;

        std::chrono::nanoseconds timer(1);
        std::this_thread::sleep_for(timer);
        
        std::cout << "Shutting down a ..." << std::endl;
    }

    std::cout << "A - Shut down complete." << std::endl;

    {
        B b;

        std::cout << "Created object b" << std::endl;

        std::chrono::nanoseconds timer(1);
        std::this_thread::sleep_for(timer);
        
        std::cout << "Shutting down b ..." << std::endl;
    }

    std::cout << "B - Shut down complete." << std::endl;

    {
        C c;

        std::cout << "Created object c" << std::endl;

        std::chrono::nanoseconds timer(1);
        std::this_thread::sleep_for(timer);
        
        std::cout << "Shutting down c ..." << std::endl;
    }

    std::cout << "C - Shut down complete." << std::endl;
    

    return 0;
}
