import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class ReadText2  {
    public static void main(String[] args)throws FileNotFoundException, IOException{
            //파일 객체 생성
            File file = new File("C:\\Users\\world\\Desktop\\javaprogramming\\FileIO\\Sample.txt");
            //입력 스트림 생성
            FileReader filereader = new FileReader(file);
            //입력 버퍼 생성
            BufferedReader bufReader = new BufferedReader(filereader);
            String line = "";
            while((line = bufReader.readLine()) != null){
                System.out.println(line);
            }
            //.readLine()은 끝에 개행문자를 읽지 않는다.            
            bufReader.close();
            // TODO: handle exception
            System.out.println("awefwaf");
    }
}

