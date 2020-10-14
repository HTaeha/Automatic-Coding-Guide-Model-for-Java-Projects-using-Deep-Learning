package com.taeha.ByteCode_Exception;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import com.taeha.support.DirExplorer;

public class BAE {
	public static void listStruct(File projectDir) throws FileNotFoundException {
		new DirExplorer((level, path, file) -> path.endsWith(".txt"), (level, path, file) -> {
			try {
				parseByteCode(file, path);
			} catch (IOException e) {
				e.printStackTrace();
			}
			count++;
		}).explore(projectDir);
	}

	public static void parseByteCode(File file, String path) throws IOException {
		String[] isException = new String[2];
		isException[0] = "throw";
		isException[1] = "exception";

		FileReader filereader = new FileReader(file);
		BufferedReader bufReader = new BufferedReader(filereader);
		String line = "";
		String line_before = "";

		String filePath = "D:\\GoogleDrive\\Research\\ByteCode\\output\\" + output + "-bytecode_exception.txt";
		File file1 = new File(filePath);
		BufferedWriter fileWrite = new BufferedWriter(new FileWriter(file1, true));

		Boolean flag = false; // method가 시작하면 true, method가 끝나면 false.
		Boolean exception_flag = false;
		// file 한 줄씩 읽기 시작.
		while (true) {
			// file의 끝에 도달하면 종료.
			if ((line = bufReader.readLine()) == null) {
				break;
			}
			if (line.equals("") || line.equals("}")) {
				flag = false;
				continue;
			}
			if (line.contains("Exception table:")) {
				fileWrite.write("EXCEPTION");
				fileWrite.newLine();
				fileWrite.flush();
				flag = false;
				continue;
			}
			if (flag) {
				exception_flag = false;
				for (int i = 0; i < isException.length; i++) {
					if (line.toLowerCase().contains(isException[i])) {
						fileWrite.write("EXCEPTION");
						fileWrite.newLine();
						fileWrite.flush();
						exception_flag = true;
					}
				}
				if (!exception_flag) {
					//try {
					if(line.trim().equals("}")) {
						fileWrite.write("\t}");
					}else {
						fileWrite.write("\t" + line.split(":")[1].replaceAll("\n|\r|\\[|\\]|\"|/|\\.|_", " ").trim());
					}
					//}catch(ArrayIndexOutOfBoundsException e) {
					//	e.printStackTrace();
					//	System.out.println(path);
					//}
					fileWrite.newLine();
					fileWrite.flush();
					continue;
				}
			}
			// Code: 가 나오면 그 윗줄이 method 이름. Code: 밑으로 method 안 내용 시작.
			if (line.contains("Code:")) {
				flag = true;

				String temp_split = line_before.split("\\(")[0];
				String[] method_split = temp_split.split(" |\\.");
				String method = method_split[method_split.length - 1];
				try {
					fileWrite.write("path: /" + path);
					fileWrite.newLine();
					fileWrite.flush();
					fileWrite.write("method: " + method);
					fileWrite.newLine();
					fileWrite.flush();

					// throws가 나온다면 exception handling을 한 것이므로 그 부분 제거 후 출력, labeling 1.
					if (line_before.contains("throws")) {
						fileWrite.write(line_before.split("throws")[0].replaceAll("\n|\r|\\[|\\]|\"|/|\\.|_", " ").trim());
						fileWrite.newLine();
						fileWrite.flush();
						fileWrite.write("EXCEPTION");
						fileWrite.newLine();
						fileWrite.flush();
					} else {
						fileWrite.write(line_before.replaceAll("\n|\r|\\[|\\]|\"|/|\\.|_", " ").trim());
						fileWrite.newLine();
						fileWrite.flush();
					}
				} catch (IOException e1) {
					e1.printStackTrace();
				}
			}
			line_before = line;
		}
		bufReader.close();
		fileWrite.close();
	}

	static String input = "guava";
	static String output = "guava";
	static int count = 0;

	public static void main(String[] args) {
		File projectDir = new File("D:\\GoogleDrive\\Research\\ByteCode\\" + input + "_disassemble_file");
		//File projectDir = new File("D:\\GoogleDrive\\Research\\JavaAutoException\\JavaParser//input//" + input + "-2.1.0");
		try {
			listStruct(projectDir);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		System.out.println(count);
	}
}
