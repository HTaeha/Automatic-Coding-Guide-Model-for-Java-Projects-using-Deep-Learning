package test;

public class java30 {
    public static void main(String[] args){
        Test test = new Test();

        try {
            test.test("1", "a");
        }catch(NumberFormatException e){
            System.out.println("input is not number...");
        }
    }
}
class Test {
    public void test(String a, String b) throws NumberFormatException{
        try{
            int sum = Integer.parseInt(a) + Integer.parseInt(b);
            System.out.println( a +"," + b +" sum ="+ sum);
        }catch(NumberFormatException e){
            System.out.println("This is not numeric character.");
            throw e;

        }
    }
}
