package com.taeha.ExtractOriginalCode;

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

				String[] isLog = new String[5];
				isLog[0] = "log";
				isLog[1] = "print";
				isLog[2] = "error";
				isLog[3] = "abort";
				isLog[4] = "assert";
				
				String[] checkList = new String[11];
				checkList[0] = "if";
				checkList[1] = "else";
				checkList[2] = "for";
				checkList[3] = "foreach";
				checkList[4] = "while";
				checkList[5] = "try";
				checkList[6] = "catch";
				checkList[7] = "finally";
				checkList[8] = "throw";
				checkList[9] = "switch";
				checkList[10] = "return";

				String filePath = "D:\\Google 드라이브\\Research\\JavaAutoLogging\\JavaParser\\output\\hbase-original_code_assert--.txt";
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
/*
					public void visit(NameExpr _name, Object arg) {
						for (i = 0; i < isLog.length; i++) {
							if (_name.getNameAsString().contains(isLog[i])) {
								try {
									fileWrite.write("LOGGER");
									fileWrite.newLine();
									fileWrite.flush();
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
						}
						super.visit(_name, arg);
					}
					public void visit(MethodCallExpr _methodCall, Object arg) {
						for (i = 0; i < isLog.length; i++) {
							if (_methodCall.getNameAsExpression().toString().contains(isLog[i])) {
								try {
									fileWrite.write("LOGGER");
									fileWrite.newLine();
									fileWrite.flush();
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
						}
						super.visit(_methodCall, arg);
					}
*/
					public void visit(MethodDeclaration _method, Object arg) {
						boolean log_flag = false;
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
	    					/*for(i = 0; i < Body_split.length; i++) {
	    						log_flag = false;
	    						for(j = 0; j < isLog.length; j++) {
		    						if(Body_split[i].contains(isLog[j])) {
		    							log_flag = true;
		    							
		    							fileWrite.write("LOGGER");fileWrite.newLine();
	    								fileWrite.flush();
		    							break;
		    						}
	    						}
	    						if(log_flag) {
	    							continue;
	    						}
	    						for(j = 0; j < checkList.length; j++) {*/
	    							//if(Body_split[i].contains(checkList[j]) && !Body_split[i].contains("/*") && !Body_split[i].contains("*/")) {
	    							/*	fileWrite.write(Body_split[i]); fileWrite.newLine();
	    								fileWrite.flush();
	    							}
	    						}
	    						
	    					}*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	    					//Original code logging method만 LOGGER로 추출.
	    					for(i = 0; i < Body_split.length; i++) {
	    						log_flag = false;
	    						
	    						for(j = 0; j < isLog.length; j++) {
		    						if(Body_split[i].contains(isLog[j])) {
		    							log_flag = true;
		    							
		    							if(Body_split[i].contains("*/")) {
		    								fileWrite.write("*/"); fileWrite.newLine();
		    								fileWrite.flush();
		    							}
		    							fileWrite.write("LOGGER");fileWrite.newLine();
	    								fileWrite.flush();
		    							break;
		    						}
	    						}
	    						if(log_flag == false){
	    							fileWrite.write(Body_split[i]); fileWrite.newLine();
	    	    					fileWrite.flush();
	    						}
	    					}
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
				"D:\\Google 드라이브\\Research\\JavaAutoLogging\\JavaParser\\input\\hbase-2.1.0");
		listStruct(projectDir);
	}
}