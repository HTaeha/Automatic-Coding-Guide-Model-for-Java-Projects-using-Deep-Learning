package test;

public class java40 {
    public static void main(String[] args){
        Test test = new Test();

            test.test("1", "a");
            System.out.println("input is not number...");
    }
}
class Test {
    public void test(String a, String b){
            int sum = Integer.parseInt(a) + Integer.parseInt(b);
            System.out.println( a +"," + b +" sum ="+ sum);
            System.out.println("This is not numeric character.");
    }
}
