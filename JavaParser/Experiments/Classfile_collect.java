package com.taeha.Classfile_collect;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import com.taeha.support.DirExplorer;

/**
 * Hello world!
 *
 */
public class App 
{
	public static void listStruct(File projectDir) throws FileNotFoundException {
		new DirExplorer((level, path, file) -> path.endsWith(".class"), (level, path, file) -> {
			System.out.println(path);
			FileCopy(path);
			count++;
		}).explore(projectDir);
	}
	public static void FileCopy(String path){      
	        FileInputStream input = null;
	        FileOutputStream output = null;
	        
	        String[] split_method = path.split("/");
	        String method = split_method[split_method.length-1];
	        try{
	            // 복사할 대상 파일을 지정해준다.
	            File file = new File("D:\\GoogleDrive\\Research\\ByteCode\\guava_class_file//"+path);
	             
	            // FileInputStream 는 File object를 생성자 인수로 받을 수 있다.         
	            input = new FileInputStream(file);
	            // 복사된 파일의 위치를 지정해준다.
	            output = new FileOutputStream(new File("D:\\GoogleDrive\\Research\\ByteCode\\guava_class_collected\\" + method));
	                         
	            int readBuffer = 0;
	            byte [] buffer = new byte[512];
	            while((readBuffer = input.read(buffer)) != -1) {
	                output.write(buffer, 0, readBuffer);
	            }
	        } catch (IOException e) {
	            System.out.println(e);
	        } finally {
	            try{
	                // 생성된 InputStream Object를 닫아준다.
	                input.close();
	                // 생성된 OutputStream Object를 닫아준다.
	                output.close();
	            } catch(IOException io) {}
	        }
	    
	}
	static int count = 0;
    public static void main( String[] args )
    {
    	File projectDir = new File("D:\\GoogleDrive\\Research\\ByteCode\\guava_class_file");
		try {
			listStruct(projectDir);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		System.out.println(count);
    }
}
