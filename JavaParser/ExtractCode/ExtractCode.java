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
import com.github.javaparser.ast.stmt.ForeachStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.ReturnStmt;
import com.github.javaparser.ast.stmt.SwitchStmt;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class ExtractCode {
	public static void main(String[] args) throws IOException {
		
		String data_type = "FN";
		String f_name = "hbase-function_AST_feature++_balanced_max400_masking_min10_ep15_1";
		String min_sentence_len = "10";
    	File file_pred = new File("D:\\Google 드라이브\\Research\\JavaAutoException\\Code_snippet\\"+data_type+"_"+f_name+".txt");
		//+data_type+"_eclipse-catch_snippet_min10_ep10_1.txt");
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
    		String[] parts = line_path.split("\\/\\/");
    		String part_last = parts[parts.length - 1];
    		String[] parts2 = part_last.split("\\.");
    		String output_name = parts2[0];
    		output_name = output_name + "-" + line_method;
    		FileInputStream input = null;
            try{
                // 복사할 대상 파일을 지정해준다.
                File input_file = new File("D:\\Google 드라이브\\Research\\JavaAutoException\\JavaParser\\input\\hbase-2.1.0\\" + line_path);
                // FileInputStream 는 File object를 생성자 인수로 받을 수 있다.         
                input = new FileInputStream(input_file);
                
                int num = 1;
                int num_t = 1;
                String filePath = "D:\\Google 드라이브\\Research\\JavaAutoException\\Code_snippet_output\\"+f_name+"\\"+data_type+"\\" + output_name + num + ".java";
                String filePath_t = "D:\\Google 드라이브\\Research\\JavaAutoException\\Code_snippet_output\\"+f_name+"\\"+data_type+"_tree\\" + output_name + num_t + ".txt";
	        	File out_file = new File(filePath);
	        	File out_file_t = new File(filePath_t);
	        	while(true) {
	        		if(out_file.exists()) {
	        			num++;
	        			filePath = "D:\\Google 드라이브\\Research\\JavaAutoException\\Code_snippet_output\\"+f_name+"\\"+data_type+"\\" + output_name + num + ".java";
	        			out_file = new File(filePath);
	        		}else {
	        			break;
	        		}
	        	}
	        	while(true) {
	        		if(out_file_t.exists()) {
	        			num_t++;
	        			filePath_t = "D:\\Google 드라이브\\Research\\JavaAutoException\\Code_snippet_output\\"+f_name+"\\"+data_type+"_tree\\" + output_name + num_t + ".txt";
		        		out_file_t = new File(filePath_t);
	        		}else {
	        			break;
	        		}
	        	}
	        	BufferedWriter fileWrite = new BufferedWriter(new FileWriter(out_file, true));
                BufferedWriter fileWrite_t = new BufferedWriter(new FileWriter(out_file_t, true));
                
                String[] isLog = new String[13];
	        	isLog[0] = "Logger";
	        	isLog[1] = "_logger";
	        	isLog[2] = "LOG";
	        	isLog[3] = "LOGGER";
	        	isLog[4] = "logger";
	        	isLog[5] = "log";
	        	isLog[6] = "METALOG";
	        	isLog[7] = "AUDITLOG";
	        	isLog[8] = "print";
				isLog[9] = "errors";
				isLog[10] = "assert";
				isLog[11] = "abort";
				isLog[12] = "showErrorHint";
	        	
				boolean isCondition = true;
				//Expression부분 자체를 지우고 해야 함. expression statement를 search하는 것 자체로 결과가 달라지기 때문에.
				boolean isExpression = true;
				
	        	CompilationUnit cu = JavaParser.parse(input_file);
                String method_name = line_method;
                cu.accept(new VoidVisitorAdapter<Object>() {
					int indent = 0;
					int i, j;
					boolean log_flag = false;

					public void visit(IfStmt _if, Object arg) {
						String ifCondition = _if.getCondition().toString();
						String[] parts = ifCondition.split("\"");
						try {
							for (int i = 0; i < indent; i++) {
								fileWrite_t.write("\t");
							}
							fileWrite_t.write(indent + "IF" + " ");
							fileWrite_t.flush();
							for (i = 0; i < parts.length; i++) {
								String[] part = parts[i].split(" ");
								for (j = 0; j < part.length; j++) {
									// if문 condition부분에 logging method가 있을 때 logging method가 추출되지 않도록 한다.
									/*
									 * for(int k=0;k<isLog.length;k++) { if(part[j].contains(isLog[k])) { log_flag =
									 * 1; break; } } if(log_flag == 1) { log_flag = 0; continue; }
									 */
									fileWrite_t.write(part[j] + " ");
									fileWrite_t.flush();
								}

							}
							fileWrite_t.newLine();
							fileWrite_t.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}

						super.visit(_if, arg);
						if (_if.hasElseBlock() && indent > 1) {
							try {
								for (int i = 0; i < indent; i++) {
									fileWrite_t.write("\t");
								}
								fileWrite_t.write(indent + "ELSE");
								fileWrite_t.newLine();
								fileWrite_t.flush();
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
					}

					public void visit(ForStmt _for, Object arg) {
						String init = _for.getInitialization().toString().replace("[", "").replace("]", "").replace("\"", "");
						String comp = _for.getCompare().toString().replace("[", "").replace("]", "").replace("Optional", "").replace("\"", "");
						String update = _for.getUpdate().toString().replace("[", "").replace("]", "").replace("\"", "");

						try {
							if (isCondition) {
								for (int i = 0; i < indent; i++) {
									fileWrite_t.write("\t");
								}
								fileWrite_t.write(indent + "FOR" + " " + init + "; " + comp + "; " + update);
								fileWrite_t.newLine();
								fileWrite_t.flush();
							}else {
								for (int i = 0; i < indent; i++) {
									fileWrite_t.write("\t");
								}
								fileWrite_t.write(indent + "FOR");
								fileWrite_t.newLine();
								fileWrite_t.flush();
							}
						} catch (IOException e) {
							e.printStackTrace();
						}

						super.visit(_for, arg);
					}

					public void visit(ForeachStmt _foreach, Object arg) {
						try {
							if(isCondition) {
								for (int i = 0; i < indent; i++) {
									fileWrite_t.write("\t");
								}
								fileWrite_t.write(
										indent + "FOR_EACH" + " " + _foreach.getVariable().toString().replace("\"", "")
												+ " : " + _foreach.getIterable().toString().replace("\"", ""));
								fileWrite_t.newLine();
								fileWrite_t.flush();
							}else {
								for (int i = 0; i < indent; i++) {
									fileWrite_t.write("\t");
								}
								fileWrite_t.write(indent + "FOR_EACH");
								fileWrite_t.newLine();
								fileWrite_t.flush();
							}
						} catch (IOException e) {
							e.printStackTrace();
						}

						super.visit(_foreach, arg);
					}

					public void visit(WhileStmt _while, Object arg) {
						try {
							for (int i = 0; i < indent; i++) {
								fileWrite_t.write("\t");
							}
							if(isCondition) {
								fileWrite_t.write(indent + "WHILE" + " " + _while.getCondition().toString().replace("\"", ""));
							}else {
								fileWrite_t.write(indent + "WHILE");
							}
							fileWrite_t.newLine();
							fileWrite_t.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}

						super.visit(_while, arg);
					}

					public void visit(TryStmt _try, Object arg) {
						try {
							for (int i = 0; i < indent; i++) {
								fileWrite_t.write("\t");
							}
							if(isCondition) {
								fileWrite_t.write(indent + "TRY" + ' ' + _try.getResources().toString().replace("[", "").replace("]","").replace("\"", ""));
							}else {
								fileWrite_t.write(indent + "TRY");
							}
							fileWrite_t.newLine();
							fileWrite_t.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}
						super.visit(_try, arg);
						if (_try.getFinallyBlock().isPresent()) {
							try {
								for (int i = 0; i < indent; i++) {
									fileWrite_t.write("\t");
								}
								fileWrite_t.write(indent + "FINALLY");
								fileWrite_t.newLine();
								fileWrite_t.flush();
							} catch (IOException e) {
								e.printStackTrace();
							}
						}

						String catchParam = _try.getCatchClauses().toString();
						if (catchParam.contains("(")) {
							String[] parts = catchParam.split("\\(");
							String part1 = parts[1];
							String[] excep = part1.split("\\)");
							if (_try.getEnd().isPresent()) {
								try {
									for (int i = 0; i < indent; i++) {
										fileWrite_t.write("\t");
									}
									fileWrite_t.write(indent + "CATCH " + excep[0]);
									fileWrite_t.newLine();
									fileWrite_t.flush();
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
						}
					}/*
						 * public void visit(CatchClause _catch, Object arg) { try { for(int
						 * i=0;i<indent;i++){ fileWrite_t.write("\t"); } fileWrite_t.write(indent + "CATCH "
						 * + _catch.getParameter()); fileWrite_t.newLine(); fileWrite_t.flush(); } catch
						 * (IOException e) { e.printStackTrace(); } super.visit(_catch, arg); }
						 */

					public void visit(ThrowStmt _throw, Object arg) {
						String throwParam = _throw.getExpression().toString().replace("\"", "");
						String[] parts = throwParam.split("\\(");  //parts[0] : exception 부분까지만 추출.
						try {
							for (int i = 0; i < indent; i++) {   
								fileWrite_t.write("\t");
							}
							//fileWrite_t.write(indent + "THROW" + " " + parts[0]);
							fileWrite_t.write(indent + "THROW" + " " + throwParam);
							fileWrite_t.newLine();
							fileWrite_t.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}

						super.visit(_throw, arg);
					}

					public void visit(SwitchStmt _switch, Object arg) {
						try {
							for (int i = 0; i < indent; i++) {
								fileWrite_t.write("\t");
							}
							if(isCondition) {
								fileWrite_t.write(indent + "SWITCH" + " " + _switch.getSelector());
							}else {
								fileWrite_t.write(indent + "SWITCH");
							}
							fileWrite_t.newLine();
							fileWrite_t.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}

						super.visit(_switch, arg);
					}
					public void visit(ReturnStmt _return, Object arg) {
						String ret = _return.getExpression().toString().replace("Optional[", "").replace("]", "").replace("\"", "");
						try {
							for (int i = 0; i < indent; i++) {
								fileWrite_t.write("\t");
							}
							fileWrite_t.write(indent + "RETURN" + " " + ret);
							fileWrite_t.newLine();
							fileWrite_t.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}

						super.visit(_return, arg);
					}

					public void visit(BlockStmt _block, Object arg) {
						indent++;
						super.visit(_block, arg);
						indent--;
					}
					
					public void visit(ExpressionStmt _exp, Object arg) {
						if (isExpression) {
							try {
								for (i = 0; i < isLog.length; i++) {
									// Logging method이거나 throw method이면 expression을 write하지 않음.
									if (_exp.toString().contains(isLog[i]) || _exp.toString().contains("throw")) {
										log_flag = true;
										break;
									}
								}
								if (!log_flag) {
									for (int i = 0; i < indent; i++) {
										fileWrite_t.write("\t");
									}
									String part = _exp.toString().split("//")[0];
									String[] parts = part.split("\"");
									fileWrite_t.write(indent + "");
									fileWrite_t.flush();
									for (j = 0; j < parts.length; j++) {
										fileWrite_t.write(parts[j] + ' ');
										fileWrite_t.flush();
									}
									fileWrite_t.newLine();
									fileWrite_t.flush();
								}
								log_flag = false;
							} catch (IOException e) {
								e.printStackTrace();
							}
							super.visit(_exp, arg);
						}
					}
					
					public void visit(NameExpr _name, Object arg) {
						if(_name.getNameAsExpression().toString().contains("throw")) {
							try {
								for (int i = 0; i < indent; i++) {
									fileWrite_t.write("\t");
								}
								fileWrite_t.write(indent + "THROW");
								fileWrite_t.newLine();
								fileWrite_t.flush();
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
						for (i = 0; i < isLog.length; i++) {
							if (_name.getNameAsString().contains(isLog[i])) {
								try {
									for (int i = 0; i < indent; i++) {
										fileWrite_t.write("\t");
									}
									fileWrite_t.write(indent + "LOGGER");
									fileWrite_t.newLine();
									fileWrite_t.flush();
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
						}
						super.visit(_name, arg);
					}
					public void visit(MethodCallExpr _methodCall, Object arg) {
						if(_methodCall.getNameAsExpression().toString().contains("throw")) {
							try {
								for (int i = 0; i < indent; i++) {
									fileWrite_t.write("\t");
								}
								fileWrite_t.write(indent + "THROW");
								fileWrite_t.newLine();
								fileWrite_t.flush();
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
						for (i = 0; i < isLog.length; i++) {
							if (_methodCall.getNameAsExpression().toString().contains(isLog[i])) {
								try {
									for (int i = 0; i < indent; i++) {
										fileWrite_t.write("\t");
									}
									fileWrite_t.write(indent + "LOGGER");
									fileWrite_t.newLine();
									fileWrite_t.flush();
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
						}
						super.visit(_methodCall, arg);
					}

					public void visit(MethodDeclaration _method, Object arg) {
						if(_method.getNameAsString().equals(method_name)) {
                			try {
								fileWrite.write(_method.getDeclarationAsString()); fileWrite.newLine();
		    					fileWrite.flush();
		    					fileWrite.write(String.valueOf(_method.getBody())); fileWrite.newLine();
		    					fileWrite.flush();
							} catch (IOException e) {
								e.printStackTrace();
							}
                			
                			try {
    							for(int i=0;i<indent;i++){
    		    					fileWrite_t.write("\t");
    		    				}
    							fileWrite_t.write(indent + _method.getDeclarationAsString().replace("\"", "")); 
    							fileWrite_t.newLine();
    							fileWrite_t.flush();
    						} catch (IOException e) {
    							e.printStackTrace();
    						}
                		}
						super.visit(_method, arg);
					}
				}, null);
                
            } catch (IOException e) {
                System.out.println(e);
            } finally {
                try{
                    // 생성된 InputStream Object를 닫아준다.
                    input.close();
                } catch(IOException io) {}
            }
    		
    	}
    	
    }
}
