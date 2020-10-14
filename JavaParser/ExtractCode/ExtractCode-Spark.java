package com.taeha.ExtractSparkCode;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.ForeachStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.SwitchStmt;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class ExtractSparkCode {
	public static void main(final String[] args) throws IOException {
		
		String data_type = "cache";
    	File file_pred = new File("C:\\Users\\COMNET5\\Google 드라이브\\2018-2학기_연구학점제\\SparkAutoCaching\\Code_snippet\\All_"+data_type+"_1");
    	FileReader filereader = new FileReader(file_pred);
    	BufferedReader bufReader = new BufferedReader(filereader);
    	String line_path = "";
    	int sum = 0;
    	while(true){
    		if((line_path = bufReader.readLine()) == null) {
    			break;
    		}
    		String[] parts = line_path.split(":");
    		int count = Integer.parseInt(parts[1]);
    		sum += count;
    		if(count == 0) {
    			continue;
    		}
    		
    		String path = parts[0];
    		String[] parts2 = path.split("\\/");
    		String output_name = parts2[parts2.length - 1];
    		String[] output_part = output_name.split("\\.");
    		output_name = output_part[0] + "-" + count + "." + output_part[1];
    		
    		FileInputStream inputStream = null;
            FileOutputStream outputStream = null;
            try{
            	String input_Path = "C:\\Users\\COMNET5\\Google 드라이브\\2018-2학기_연구학점제\\SparkAutoCaching\\JavaParser\\input\\" + path;
            	String output_Path = "C:\\Users\\COMNET5\\Google 드라이브\\2018-2학기_연구학점제\\SparkAutoCaching\\Code_snippet_output\\"+data_type+"\\"+output_name;
            	
                inputStream = new FileInputStream(input_Path);
                outputStream = new FileOutputStream(output_Path);
                
                byte[] readBuffer = new byte[1024];
                while (inputStream.read(readBuffer, 0, readBuffer.length) != -1) {
                	outputStream.write(readBuffer);
                }
                
            } catch (IOException e) {
                System.out.println(e);
            } finally {
                try{
                    // 생성된 InputStream Object를 닫아준다.
                	inputStream.close();
                	outputStream.close();
                } catch(IOException io) {}
            }
    		
    	}
    	System.out.println(sum);
    	
    }
}
