package com.taeha.ExtractOriginalCode_TryCatching;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.type.UnionType;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.printer.PrettyPrinterConfiguration.IndentType;
import com.github.javaparser.symbolsolver.javaparsermodel.contexts.CatchClauseContext;
import com.github.javaparser.utils.CodeGenerationUtils;
import com.github.javaparser.utils.SourceRoot;
import com.google.common.base.Strings;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.taeha.support.DirExplorer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;

public class ListStructure {

	public static void listStruct(File projectDir) {
		new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
			try {
				CompilationUnit cu = JavaParser.parse(file);	

				String[] isTryCatch = new String[3];
				isTryCatch[0] = "try";
				isTryCatch[1] = "catch";
				isTryCatch[2] = "finally";
				
				String[] checkList = new String[8];
				checkList[0] = "if";
				checkList[1] = "else";
				checkList[2] = "for";
				checkList[3] = "foreach";
				checkList[4] = "while";
				checkList[5] = "throw";
				checkList[6] = "switch";
				checkList[7] = "return";				

				String filePath = "D:\\Google 드라이브\\Research\\JavaAutoException\\JavaParser\\output\\guava-exception_original_code_structure-new.txt";
				File file1 = new File(filePath);
				BufferedWriter fileWrite = new BufferedWriter(new FileWriter(file1, true));
				cu.accept(new VoidVisitorAdapter<Object>() {
					int indent = 0;
					int i, j;

					
					public void visit(BlockStmt _block, Object arg) {
						indent++;
						super.visit(_block, arg);
						indent--;
					}

					public void visit(MethodDeclaration _method, Object arg) {
						boolean try_flag = false;
						try {
							if (indent == 0) {
								fileWrite.write("path: " + path);
								fileWrite.newLine();
								fileWrite.flush();
								fileWrite.write("method: " + _method.getNameAsString());
								fileWrite.newLine();
								fileWrite.flush();
							}
							fileWrite.write(_method.getDeclarationAsString()); fileWrite.newLine();
	    					fileWrite.flush();
	    					String Body = String.valueOf(_method.getBody());
	    					String[] Body_split = Body.split("\r\n");
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	    					//Original 코드에서 structure만 남김.
							for (i = 0; i < Body_split.length; i++) {
								try_flag = false;
								for (j = 0; j < checkList.length; j++) {
									if (Body_split[i].contains(checkList[j]) && !Body_split[i].contains("/*")
											&& !Body_split[i].contains("*/")) {
										fileWrite.write(Body_split[i]);
										fileWrite.newLine();
										fileWrite.flush();
									}
								}
								for (j = 0; j < isTryCatch.length; j++) {
									if (Body_split[i].contains(isTryCatch[j])) {
										try_flag = true;

										fileWrite.write(isTryCatch[j].toUpperCase());
										fileWrite.newLine();
										fileWrite.flush();
										break;
									}
								}
							}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
							//Original code exception method만 TRY,CATCH,FINALLY로 추출.
	    					/*for(i = 0; i < Body_split.length; i++) {
	    						try_flag = false;
	    						for(j = 0; j < isTryCatch.length; j++) {
		    						if(Body_split[i].contains(isTryCatch[j])) {
		    							try_flag = true;*/
		    							//if(Body_split[i].contains("*/")) {
		    							//	fileWrite.write("*/"); fileWrite.newLine();
		    							/*	fileWrite.flush();
		    							}
		    							fileWrite.write(isTryCatch[j].toUpperCase()); fileWrite.newLine();
	    								fileWrite.flush();
		    							break;
		    						}
	    						}
	    						if(try_flag == false){
	    							fileWrite.write(Body_split[i]); fileWrite.newLine();
	    	    					fileWrite.flush();
	    						}
	    					}*/
						} catch (IOException e) {
							e.printStackTrace();
						}
						super.visit(_method, arg);
					}
				}, null);
			} catch (Exception e1) {
				e1.printStackTrace();
			}
		}).explore(projectDir);

	}

	public static void main(String[] args) throws IOException {
		File projectDir = new File(
				"D:\\Google 드라이브\\Research\\JavaAutoException\\JavaParser\\input\\guava-master");
		listStruct(projectDir);
	}
}