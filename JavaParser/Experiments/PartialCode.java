package com.taeha.MakePartialCode;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;
import com.taeha.support.DirExplorer;

public class PartialCode {

	public static void main(String[] args) throws FileNotFoundException, ParseException {
		File projectDir = new File(
				"D:\\Google 드라이브\\Research\\JavaAutoException\\JavaParser\\input\\partial_code\\" + input_dir);
		listStruct(projectDir);
	}

	static String input_dir = "hbase-2.1.0";

	public static void listStruct(File projectDir) {
		new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
			try {
				CompilationUnit parsed = JavaParser.parse(file);

				String[] parts = path.split("\\/");
				String output_dir = "";
				for (int i = 0; i < parts.length - 1; i++) {
					output_dir += "\\" + parts[i];
				}
				String part_last = parts[parts.length - 1];
				String[] parts2 = part_last.split("\\.");
				String output_name = parts2[0] + "-partial_code.java";

				String filePath = "D:\\Google 드라이브\\Research\\JavaAutoException\\JavaParser\\input\\partial_code\\"
						+ input_dir + output_dir + "\\" + output_name;
				File file1 = new File(filePath);

				String[] checkList = new String[7];
				checkList[0] = "log";
				checkList[1] = "print";
				checkList[2] = "error";
				checkList[3] = "abort";
				checkList[4] = "assert";
				checkList[5] = "throw";
				checkList[6] = "exception";
				try {
					BufferedWriter fileWrite = new BufferedWriter(new FileWriter(file1, true));

					parsed.accept(new ModifierVisitor<Void>() {
						@Override/*
						public Visitable visit(MethodCallExpr n, Void arg) {
							for (int i = 0; i < checkList.length; i++) {
								// Logging method이거나 throw method이면 expression을 delete하지 않음.
								if (n.getNameAsExpression().toString().toLowerCase().contains(checkList[i])
										|| n.getNameAsExpression().toString().toLowerCase().contains("throw")) {
									return super.visit(n, arg);
								}
							}
							return null;
						}
						public Visitable visit(VariableDeclarationExpr n, Void arg) {
							for (int i = 0; i < checkList.length; i++) {
								// Logging method이면 expression을 delete하지 않음.
								if (n.getVariables().toString().toLowerCase().replaceAll("\n|\r|\\[|\\]|\"", " ").contains(checkList[i])|| 
										n.getCommonType().toString().toLowerCase().replaceAll("\n|\r|\\[|\\]|\"", " ").contains(checkList[i])) {
									return super.visit(n, arg);
								}
							}
							return null;
						}*/
						public Visitable visit(FieldDeclaration n, Void arg) {
							for (int i = 0; i < checkList.length; i++) {
								// Logging method이면 expression을 delete하지 않음.
								if (n.getVariables().toString().toLowerCase().replaceAll("\n|\r|\\[|\\]|\"", " ").contains(checkList[i])||
										n.getModifiers().toString().toLowerCase().replaceAll("\n|\r|\\[|\\]|\"", " ").contains(checkList[i])) {
									return super.visit(n, arg);
								}
							}
							return null;
						}
						public Visitable visit(ExpressionStmt n, Void arg) {
							for (int i = 0; i < checkList.length; i++) {
								// Logging method이거나 throw method이면 expression을 delete하지 않음.
								if (n.toString().toLowerCase().contains(checkList[i]) || n.toString().toLowerCase().contains("throw")) {
									return super.visit(n, arg);
								}
							}
							return null;
						}
					}, null);

					fileWrite.write(parsed.toString());
					fileWrite.flush();
					fileWrite.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}).explore(projectDir);
	}
}