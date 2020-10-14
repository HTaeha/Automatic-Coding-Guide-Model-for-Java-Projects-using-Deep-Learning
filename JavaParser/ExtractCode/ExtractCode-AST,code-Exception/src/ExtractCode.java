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
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.ReturnStmt;
import com.github.javaparser.ast.stmt.SwitchStmt;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class ExtractCode {
	public static void main(String[] args) throws IOException {
		
		String data_type = "ASTT_codeT_err";
		String f_name = "analysis-hbase_AST_code_min10_ep15_1";
		String storage = "hbase-2.1.0\\";
		String output_f_name = "hbase-AST,code_err";
    	File file_pred = new File("D:\\Google 드라이브\\Research\\JavaAutoException\\Code_snippet\\AST,code_err\\"+data_type+"_"+f_name+".txt");
    	FileReader filereader = new FileReader(file_pred);
    	BufferedReader bufReader = new BufferedReader(filereader);
    	String line_path = "";
    	String line_method = "";
    	while(true){
    		if((line_path = bufReader.readLine()) == null) {
    			break;
    		}
    		if((line_method = bufReader.readLine()) == null) {
    			break;
    		}
    		if(bufReader.readLine() == null) {
    			break;
    		}
    		if(bufReader.readLine() == null) {
    			break;
    		}
    		if(bufReader.readLine() == null) {
    			break;
    		}
    		if(bufReader.readLine() == null) {
    			break;
    		}
    		String[] parts = line_path.split("\\/\\/");
    		String part_last = parts[parts.length - 1];
    		String[] parts2 = part_last.split("\\.");
    		String output_name = parts2[0];
    		output_name = output_name + "-" + line_method;
    		//FileInputStream input = null;
            try{
                // 복사할 대상 파일을 지정해준다.
                File input_file = new File("D:\\Google 드라이브\\Research\\JavaAutoException\\JavaParser\\input\\"+storage + line_path);
                // FileInputStream 는 File object를 생성자 인수로 받을 수 있다.         
                //input = new FileInputStream(input_file);
                
                int num = 1;
                int num_t = 1;
                String filePath = "D:\\Google 드라이브\\Research\\JavaAutoException\\Code_snippet_output\\"+output_f_name+"\\"+data_type+"\\" + output_name + num + ".java";
                String filePath_AST = "D:\\Google 드라이브\\Research\\JavaAutoException\\Code_snippet_output\\"+output_f_name+"\\"+data_type+"_tree\\" + output_name + num_t + ".txt";
	        	File out_file = new File(filePath);
	        	File out_file_AST = new File(filePath_AST);
	        	while(true) {
	        		if(out_file.exists()) {
	        			num++;
	        			filePath = "D:\\Google 드라이브\\Research\\JavaAutoException\\Code_snippet_output\\"+output_f_name+"\\"+data_type+"\\" + output_name + num + ".java";
	        			out_file = new File(filePath);
	        		}else {
	        			break;
	        		}
	        	}
	        	while(true) {
	        		if(out_file_AST.exists()) {
	        			num_t++;
	        			filePath_AST = "D:\\Google 드라이브\\Research\\JavaAutoException\\Code_snippet_output\\"+output_f_name+"\\"+data_type+"_tree\\" + output_name + num_t + ".txt";
		        		out_file_AST = new File(filePath_AST);
	        		}else {
	        			break;
	        		}
	        	}
	        	
	        	BufferedWriter fileWrite = new BufferedWriter(new FileWriter(out_file, true));
                BufferedWriter fileWrite_AST = new BufferedWriter(new FileWriter(out_file_AST, true));
                
                
                JAE.print(input_file, line_path, fileWrite, fileWrite_AST, line_method);
                
				fileWrite.close();
			} catch (IOException e) {
                System.out.println(e);
            }
    	}
    	bufReader.close();
    	
    }
}
