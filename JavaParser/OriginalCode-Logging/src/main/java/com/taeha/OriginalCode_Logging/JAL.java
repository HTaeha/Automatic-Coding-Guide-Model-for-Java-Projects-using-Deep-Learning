package com.taeha.OriginalCode_Logging;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Optional;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.ArrayCreationLevel;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.AnnotationDeclaration;
import com.github.javaparser.ast.body.AnnotationMemberDeclaration;
import com.github.javaparser.ast.body.BodyDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.EnumConstantDeclaration;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.InitializerDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.ReceiverParameter;
import com.github.javaparser.ast.body.TypeDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.comments.BlockComment;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.comments.JavadocComment;
import com.github.javaparser.ast.comments.LineComment;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.ArrayAccessExpr;
import com.github.javaparser.ast.expr.ArrayCreationExpr;
import com.github.javaparser.ast.expr.ArrayInitializerExpr;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.BooleanLiteralExpr;
import com.github.javaparser.ast.expr.CastExpr;
import com.github.javaparser.ast.expr.CharLiteralExpr;
import com.github.javaparser.ast.expr.ClassExpr;
import com.github.javaparser.ast.expr.ConditionalExpr;
import com.github.javaparser.ast.expr.DoubleLiteralExpr;
import com.github.javaparser.ast.expr.EnclosedExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.FieldAccessExpr;
import com.github.javaparser.ast.expr.InstanceOfExpr;
import com.github.javaparser.ast.expr.IntegerLiteralExpr;
import com.github.javaparser.ast.expr.LambdaExpr;
import com.github.javaparser.ast.expr.LongLiteralExpr;
import com.github.javaparser.ast.expr.MarkerAnnotationExpr;
import com.github.javaparser.ast.expr.MemberValuePair;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.MethodReferenceExpr;
import com.github.javaparser.ast.expr.Name;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.NormalAnnotationExpr;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.expr.SingleMemberAnnotationExpr;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.expr.SuperExpr;
import com.github.javaparser.ast.expr.SwitchExpr;
import com.github.javaparser.ast.expr.ThisExpr;
import com.github.javaparser.ast.expr.TypeExpr;
import com.github.javaparser.ast.expr.UnaryExpr;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.modules.ModuleDeclaration;
import com.github.javaparser.ast.modules.ModuleExportsDirective;
import com.github.javaparser.ast.modules.ModuleOpensDirective;
import com.github.javaparser.ast.modules.ModuleProvidesDirective;
import com.github.javaparser.ast.modules.ModuleRequiresDirective;
import com.github.javaparser.ast.modules.ModuleUsesDirective;
import com.github.javaparser.ast.nodeTypes.NodeWithAnnotations;
import com.github.javaparser.ast.stmt.AssertStmt;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.BreakStmt;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.stmt.ContinueStmt;
import com.github.javaparser.ast.stmt.DoStmt;
import com.github.javaparser.ast.stmt.EmptyStmt;
import com.github.javaparser.ast.stmt.ExplicitConstructorInvocationStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.ForEachStmt;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.LabeledStmt;
import com.github.javaparser.ast.stmt.LocalClassDeclarationStmt;
import com.github.javaparser.ast.stmt.ReturnStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.stmt.SwitchEntryStmt;
import com.github.javaparser.ast.stmt.SwitchStmt;
import com.github.javaparser.ast.stmt.SynchronizedStmt;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.stmt.UnparsableStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.type.ArrayType;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.IntersectionType;
import com.github.javaparser.ast.type.PrimitiveType;
import com.github.javaparser.ast.type.ReferenceType;
import com.github.javaparser.ast.type.TypeParameter;
import com.github.javaparser.ast.type.UnionType;
import com.github.javaparser.ast.type.UnknownType;
import com.github.javaparser.ast.type.VarType;
import com.github.javaparser.ast.type.VoidType;
import com.github.javaparser.ast.type.WildcardType;
import com.github.javaparser.ast.visitor.VoidVisitor;
import com.taeha.support.DirExplorer;

public class JAL {
	public static void listStruct(File projectDir) throws FileNotFoundException {
		new DirExplorer((level, path, file) -> path.endsWith(".java"), (level, path, file) -> {
			print(file, path);
		}).explore(projectDir);
	}
	
	/**
	 * @effects Parse <tt>code</tt> and print its abstract syntax tree (AST) out on
	 *          the standard output.
	 * 
	 * @author adapted from code at
	 *         {@link https://github.com/javaparser/javaparser/issues/538#issuecomment-276155353}
	 *         by Danny van Bruggen (matozoid)
	 */
	public static void print(File file, String path) {
		CompilationUnit cu = null;
		try {
			cu = JavaParser.parse(file);
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}

		String[] isLog = new String[5];
		isLog[0] = "log";
		isLog[1] = "print";
		isLog[2] = "error";
		isLog[3] = "abort";
		isLog[4] = "assert";
		
		String[] checkList = new String[6];
		checkList[0] = "if";
		checkList[1] = "for";
		checkList[2] = "foreach";
		checkList[3] = "while";
		checkList[4] = "try";
		checkList[5] = "switch";

		String filePath = "G:\\내 드라이브\\Taeha\\Research\\JavaAutoLogging\\JavaParser\\suin\\output\\" + output;
		File file1 = new File(filePath);
		try {
			BufferedWriter fileWrite = new BufferedWriter(new FileWriter(file1, true));

			cu.accept(new VoidVisitor<Integer>() {
				boolean log_flag = false;
				
				public void out(Node n, int indentLevel) {
					printNode(n, indentLevel);
				}
				public void visit(final AnnotationDeclaration n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					n.getName().accept(this, arg + 1);
					if (n.getMembers() != null) {
						for (final BodyDeclaration<?> member : n.getMembers()) {
							member.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final AnnotationMemberDeclaration n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					n.getType().accept(this, arg + 1);
					n.getDefaultValue().ifPresent(d -> d.accept(this, arg + 1));
				}

				@Override
				public void visit(final ArrayAccessExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getName().accept(this, arg + 1);
					n.getIndex().accept(this, arg + 1);
				}

				@Override
				public void visit(final ArrayCreationExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getElementType().accept(this, arg + 1);
					for (ArrayCreationLevel level : n.getLevels()) {
						level.accept(this, arg + 1);
					}
					n.getInitializer().ifPresent(i -> i.accept(this, arg + 1));
				}

				@Override
				public void visit(final ArrayInitializerExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					if (n.getValues() != null) {
						for (final Expression expr : n.getValues()) {
							expr.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final AssertStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getCheck().accept(this, arg + 1);
					n.getMessage().ifPresent(m -> m.accept(this, arg + 1));
				}

				@Override
				public void visit(final AssignExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getTarget().accept(this, arg + 1);
					n.getValue().accept(this, arg + 1);
				}

				@Override
				public void visit(final BinaryExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getLeft().accept(this, arg + 1);
					n.getRight().accept(this, arg + 1);
				}

				@Override
				public void visit(final BlockComment n, final Integer arg) {
					out(n, arg);
				}

				@Override
				public void visit(final BlockStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					if (n.getStatements() != null) {
						for (final Statement s : n.getStatements()) {
							s.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final BooleanLiteralExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
				}

				@Override
				public void visit(final BreakStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
				}

				@Override
				public void visit(final CastExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getType().accept(this, arg + 1);
					n.getExpression().accept(this, arg + 1);
				}

				@Override
				public void visit(final CatchClause n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getParameter().accept(this, arg + 1);
					n.getBody().accept(this, arg + 1);
				}

				@Override
				public void visit(final CharLiteralExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
				}

				@Override
				public void visit(final ClassExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getType().accept(this, arg + 1);
				}

				@Override
				public void visit(final ClassOrInterfaceDeclaration n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					n.getName().accept(this, arg + 1);
					for (final TypeParameter t : n.getTypeParameters()) {
						t.accept(this, arg + 1);
					}
					for (final ClassOrInterfaceType c : n.getExtendedTypes()) {
						c.accept(this, arg + 1);
					}
					for (final ClassOrInterfaceType c : n.getImplementedTypes()) {
						c.accept(this, arg + 1);
					}
					for (final BodyDeclaration<?> member : n.getMembers()) {
						member.accept(this, arg + 1);
					}
				}

				@Override
				public void visit(final ClassOrInterfaceType n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					// n.getScope().ifPresent(s -> s.accept(this, arg + 1));
					n.getTypeArguments().ifPresent(tas -> tas.forEach(ta -> ta.accept(this, arg + 1)));
				}

				@Override
				public void visit(final CompilationUnit n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					// n.getPackage().ifPresent(p -> p.accept(this, arg + 1));
					if (n.getImports() != null) {
						for (final com.github.javaparser.ast.ImportDeclaration i : n.getImports()) {
							i.accept(this, arg + 1);
						}
					}
					if (n.getTypes() != null) {
						for (final TypeDeclaration<?> typeDeclaration : n.getTypes()) {
							typeDeclaration.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final ConditionalExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getCondition().accept(this, arg + 1);
					n.getThenExpr().accept(this, arg + 1);
					n.getElseExpr().accept(this, arg + 1);
				}

				@Override
				public void visit(final ConstructorDeclaration n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					if (n.getTypeParameters() != null) {
						for (final TypeParameter t : n.getTypeParameters()) {
							t.accept(this, arg + 1);
						}
					}
					n.getName().accept(this, arg + 1);
					if (n.getParameters() != null) {
						for (final Parameter p : n.getParameters()) {
							p.accept(this, arg + 1);
						}
					}
					/*
					 * if (n.getThrows() != null) { for (final ReferenceType name : n.getThrows()) {
					 * name.accept(this, arg + 1); } }
					 */
					n.getBody().accept(this, arg + 1);
				}

				@Override
				public void visit(final ContinueStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
				}

				@Override
				public void visit(final DoStmt n, final Integer arg) {
					out(n, arg);

					visitComment(n.getComment(), arg + 1);
					n.getBody().accept(this, arg + 1);
					n.getCondition().accept(this, arg + 1);
				}

				@Override
				public void visit(final DoubleLiteralExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
				}
				/*
				 * @Override public void visit(final EmptyMemberDeclaration n, final Integer
				 * arg) {  visitComment(n.getComment(), arg + 1); }
				 */

				@Override
				public void visit(final EmptyStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
				}
				/*
				 * @Override public void visit(final EmptyTypeDeclaration n, final Integer arg)
				 * {  visitComment(n.getComment(), arg + 1);
				 * n.getNameExpr().accept(this, arg + 1); }
				 */

				@Override
				public void visit(final EnclosedExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					// n.getInner().ifPresent(i -> i.accept(this, arg + 1));
				}

				@Override
				public void visit(final EnumConstantDeclaration n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					if (n.getArguments() != null) {
						for (final Expression e : n.getArguments()) {
							e.accept(this, arg + 1);
						}
					}
					if (n.getClassBody() != null) {
						for (final BodyDeclaration<?> member : n.getClassBody()) {
							member.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final EnumDeclaration n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					n.getName().accept(this, arg + 1);
					if (n.getImplementedTypes() != null) {
						for (final ClassOrInterfaceType c : n.getImplementedTypes()) {
							c.accept(this, arg + 1);
						}
					}
					if (n.getEntries() != null) {
						for (final EnumConstantDeclaration e : n.getEntries()) {
							e.accept(this, arg + 1);
						}
					}
					if (n.getMembers() != null) {
						for (final BodyDeclaration<?> member : n.getMembers()) {
							member.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final ExplicitConstructorInvocationStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					if (!n.isThis() && n.getExpression().isPresent()) {
						n.getExpression().get().accept(this, arg + 1);
					}
					n.getTypeArguments().ifPresent(tas -> tas.forEach(ta -> ta.accept(this, arg + 1)));
					if (n.getArguments() != null) {
						for (final Expression e : n.getArguments()) {
							e.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final ExpressionStmt n, final Integer arg) {
					log_flag = false;
					out(n, arg);
					write_logged_method(n.toString(),n, arg);
					
					if (!log_flag) {
						visitComment(n.getComment(), arg + 1);
						n.getExpression().accept(this, arg + 1);
					}
				}

				@Override
				public void visit(final FieldAccessExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getScope().accept(this, arg + 1);
					n.getField().accept(this, arg + 1);
				}

				@Override
				public void visit(final FieldDeclaration n, final Integer arg) {
					log_flag = false;
					out(n, arg);
					write_logged_method(n.getVariables().toString(), n,arg);
					write_logged_method(n.getModifiers().toString(),n, arg);

					/*
					 * visitComment(n.getComment(), arg + 1); visitAnnotations(n, arg + 1);
					 * n.getElementType().accept(this, arg + 1); for (final VariableDeclarator var :
					 * n.getVariables()) { var.accept(this, arg + 1); }
					 */
				}

				@Override
				public void visit(final ForEachStmt n, final Integer arg) {
					out(n, arg);

					visitComment(n.getComment(), arg + 1);
					n.getVariable().accept(this, arg + 1);
					n.getIterable().accept(this, arg + 1);
					n.getBody().accept(this, arg + 1);
				}

				@Override
				public void visit(final ForStmt n, final Integer arg) {
					out(n, arg);

					visitComment(n.getComment(), arg + 1);
					/*
					 * for (final Expression e : n.getInit()) { e.accept(this, arg + 1); }
					 */
					n.getCompare().ifPresent(c -> c.accept(this, arg + 1));
					for (final Expression e : n.getUpdate()) {
						e.accept(this, arg + 1);
					}
					n.getBody().accept(this, arg + 1);
				}

				@Override
				public void visit(final IfStmt n, final Integer arg) {
					out(n, arg);

					visitComment(n.getComment(), arg + 1);
					n.getCondition().accept(this, arg + 1);
					n.getThenStmt().accept(this, arg + 1);
					n.getElseStmt().ifPresent(es -> es.accept(this, arg + 1));
				}

				@Override
				public void visit(final InitializerDeclaration n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					// n.getBlock().accept(this, arg + 1);
				}

				@Override
				public void visit(final InstanceOfExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getExpression().accept(this, arg + 1);
					n.getType().accept(this, arg + 1);
				}

				@Override
				public void visit(final IntegerLiteralExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
				}
				/*
				 * @Override public void visit(final IntegerLiteralMinValueExpr n, final Integer
				 * arg) {  visitComment(n.getComment(), arg + 1); }
				 */

				@Override
				public void visit(final JavadocComment n, final Integer arg) {
					out(n, arg);
				}

				@Override
				public void visit(final LabeledStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getStatement().accept(this, arg + 1);
				}

				@Override
				public void visit(final LineComment n, final Integer arg) {
					out(n, arg);
				}

				@Override
				public void visit(final LongLiteralExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
				}
				/*
				 * @Override public void visit(final LongLiteralMinValueExpr n, final Integer
				 * arg) {  visitComment(n.getComment(), arg + 1); }
				 */

				@Override
				public void visit(final MarkerAnnotationExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getName().accept(this, arg + 1);
				}

				@Override
				public void visit(final MemberValuePair n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getValue().accept(this, arg + 1);
				}

				@Override
				public void visit(final MethodCallExpr n, final Integer arg) {
					log_flag = false;
					out(n, arg);
					write_logged_method(n.getNameAsExpression().toString(), n,arg);
					if (!log_flag) {
						visitComment(n.getComment(), arg + 1);
						n.getScope().ifPresent(s -> s.accept(this, arg + 1));
						n.getTypeArguments().ifPresent(tas -> tas.forEach(ta -> ta.accept(this, arg + 1)));
						n.getName().accept(this, arg + 1);
						if (n.getArguments() != null) {
							for (final Expression e : n.getArguments()) {
								e.accept(this, arg + 1);
							}
						}
					}
				}

				@Override
				public void visit(final MethodDeclaration n, final Integer arg) {
					if (arg == 2) {
						try {
							fileWrite.newLine();
							fileWrite.write("path: " + path);
							fileWrite.newLine();
							fileWrite.flush();
							fileWrite.write("method: " + n.getNameAsString());
							fileWrite.flush();
						} catch (IOException e1) {
							e1.printStackTrace();
						}
					}
					out(n, arg);
						
					if(arg == 2) {
						try {
							fileWrite.newLine();
							fileWrite.write(n.getDeclarationAsString().split(" ")[0] + " ");
							fileWrite.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}
						String[] parts = n.getTypeAsString().toString().replaceAll("\n|\r|\\[|\\]|\"", " ").split(" ");
						log_flag = false;
						for (int j = 0; j < parts.length; j++) {

							for (int i = 0; i < isLog.length; i++) {
								if (parts[j].toLowerCase().contains(isLog[i])) {
									log_flag = true;
									break;
								}
								if (i == isLog.length - 1 && !log_flag) {
									try {
										fileWrite.write(parts[j] + " ");
										fileWrite.flush();
									} catch (IOException e) {
										e.printStackTrace();
									}
								}
							}
							log_flag = false;
						}
						try {
							fileWrite.write(n.getNameAsString() + " ");
							fileWrite.flush();
						} catch (IOException e1) {
							e1.printStackTrace();
						}
						String[] parts1 = n.getParameters().toString().replaceAll("\n|\r|\\[|\\]|\"", " ").split(",");

						for (int j = 0; j < parts1.length; j++) {
							for (int i = 0; i < isLog.length; i++) {
								if (parts1[j].toLowerCase().contains(isLog[i])) {
									log_flag = true;
									break;
								}
								if (i == isLog.length - 1 && !log_flag) {
									try {
										fileWrite.write(parts1[j] + " ");
										fileWrite.flush();
									} catch (IOException e) {
										e.printStackTrace();
									}
								}
							}
							log_flag = false;
						}
						try {
							if (n.getDeclarationAsString().contains("throws")) {
								fileWrite.write(" throws " + n.getDeclarationAsString().split("throws")[1]);
							}
							fileWrite.newLine();
							fileWrite.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}

						String Body = String.valueOf(n.getBody());
						String[] Body_split = Body.split("\r\n");

						boolean isControlStmt = false;
						boolean annotationFlag = false;
						int CS_index;
						for (int i = 0; i < Body_split.length; i++) {
							log_flag = false;
							char[] str_body = Body_split[i].toCharArray();
							/*
							 * for(int j = 0;j < str_body.length; j++) { if(str_body[i] == '/' &&
							 * str_body[i+1] == '/') { str_body[i] = 0; } }
							 */
							if (Body_split[i].contains("//")) {
								continue;
							} else if (Body_split[i].contains("\"/*\"") || Body_split[i].contains("/*\"")
									|| Body_split[i].contains("\"/*")) {
								;
							} else if (Body_split[i].contains("/*") && Body_split[i].contains("*/")) {
								continue;
							} else if (Body_split[i].contains("/*")) {
								annotationFlag = true;
								continue;
							} else if (Body_split[i].contains("*/")) {
								annotationFlag = false;
								continue;
							}
							if (annotationFlag == true) {
								continue;
							}
							for (int h = 0; h < checkList.length; h++) {
								// Body_split이 checkList(control statement)를 포함하면 Logging method를 검사하지 않고 출력한다.
								if ((CS_index = Body_split[i].indexOf(checkList[h])) != -1) {
									//checkList(control statement)의 바로 앞 character가 space, tab이고 
									//다음 char가 space, tab, (, { 중 하나이면 control statement라고 판별한다.
									if (checkStat(CS_index, str_body, checkList[h].length()) && Body_split[i].contains("{")) {
										try {
											fileWrite.write(Body_split[i]);
											fileWrite.newLine();
											fileWrite.flush();
											isControlStmt = true;
											break;
										} catch (IOException e) {
											e.printStackTrace();
										}
									}
								}
							}
							if (isControlStmt) {
								isControlStmt = false;
								continue;
							}
							//remove Logging method.
							
							String[] temp = Body_split[i].split(" ");
							for (int k = 0; k < temp.length; k++) {
								for (int j = 0; j < isLog.length; j++) {
									if (temp[k].toLowerCase().contains(isLog[j])) {
										log_flag = true;
										break;
									}
								}
								if (log_flag == false) {
									try {
										fileWrite.write(temp[k] + " ");
										fileWrite.flush();
									} catch (IOException e) {
										e.printStackTrace();
									}
								}
							}/*
							for (int j = 0; j < isLog.length; j++) {
								if (Body_split[i].toLowerCase().contains(isLog[j])) {
									log_flag = true;
									break;
								}
							}
							if (!log_flag) {
								try {
									fileWrite.write(Body_split[i] + " ");
									fileWrite.flush();
								} catch (IOException e) {
									e.printStackTrace();
								}
							}*/
							try {
								fileWrite.newLine();
								fileWrite.flush();
							} catch (IOException e) {
								e.printStackTrace();
							}

						}
					}

					n.getBody().ifPresent(b -> b.accept(this, arg + 1));
				}
				private boolean checkStat(int idx, char[] arr, int len) {
					if(leftCheck(arr[idx-1]) || idx == 0) {
						if(rightCheck(arr[idx+len])) {
							return true;
						}
					}
					return false;
				}
				private boolean leftCheck(char ch) {
					if(ch == '\t' || ch ==' ') {
						return true;
					}else {
						return false;
					}
				}
				private boolean rightCheck(char ch) {
					if(ch == '\t' || ch ==' '|| ch =='('|| ch =='{') {
						return true;
					}else {
						return false;
					}
				}
				@Override
				public void visit(final NameExpr n, final Integer arg) {
					log_flag = false;
					out(n, arg);
					write_logged_method(n.getNameAsExpression().toString(),n, arg);
					if(!log_flag) {
						visitComment(n.getComment(), arg + 1);
					}
				}

				@Override
				public void visit(final NormalAnnotationExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getName().accept(this, arg + 1);
					if (n.getPairs() != null) {
						for (final MemberValuePair m : n.getPairs()) {
							m.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final NullLiteralExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
				}

				@Override
				public void visit(final ObjectCreationExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getScope().ifPresent(s -> s.accept(this, arg + 1));
					n.getTypeArguments().ifPresent(tas -> tas.forEach(ta -> ta.accept(this, arg + 1)));
					n.getType().accept(this, arg + 1);
					if (n.getArguments() != null) {
						for (final Expression e : n.getArguments()) {
							e.accept(this, arg + 1);
						}
					}
					n.getAnonymousClassBody().ifPresent(acb -> acb.forEach(m -> m.accept(this, arg + 1)));
				}

				@Override
				public void visit(final PackageDeclaration n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					n.getName().accept(this, arg + 1);
				}

				@Override
				public void visit(final Parameter n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					n.getType().accept(this, arg + 1);
					// n.getId().accept(this, arg + 1);
				}

				@Override
				public void visit(final PrimitiveType n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
				}
				/*
				 * @Override public void visit(final QualifiedNameExpr n, final Integer arg) {
				 *  visitComment(n.getComment(), arg + 1);
				 * n.getQualifier().accept(this, arg + 1); }
				 */

				@Override
				public void visit(ArrayType n, Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					n.getComponentType().accept(this, arg + 1);
				}

				@Override
				public void visit(ArrayCreationLevel n, Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					n.getDimension().ifPresent(d -> d.accept(this, arg + 1));
				}

				@Override
				public void visit(final IntersectionType n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					for (ReferenceType element : n.getElements()) {
						element.accept(this, arg + 1);
					}
				}

				@Override
				public void visit(final UnionType n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					for (ReferenceType element : n.getElements()) {
						element.accept(this, arg + 1);
					}
				}

				@Override
				public void visit(final ReturnStmt n, final Integer arg) {
					log_flag = false;
					out(n, arg);
					write_logged_method(n.getExpression().toString(),n, arg);				
					if (!log_flag) {
						visitComment(n.getComment(), arg + 1);
						n.getExpression().ifPresent(e -> e.accept(this, arg + 1));
					}
				}

				@Override
				public void visit(final SingleMemberAnnotationExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getName().accept(this, arg + 1);
					n.getMemberValue().accept(this, arg + 1);
				}

				@Override
				public void visit(final StringLiteralExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
				}

				@Override
				public void visit(final SuperExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getClassExpr().ifPresent(ce -> ce.accept(this, arg + 1));
				}

				@Override
				public void visit(final SwitchEntryStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getLabel().ifPresent(l -> l.accept(this, arg + 1));
					if (n.getStatements() != null) {
						for (final Statement s : n.getStatements()) {
							s.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final SwitchStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getSelector().accept(this, arg + 1);
					if (n.getEntries() != null) {
						for (final SwitchEntryStmt e : n.getEntries()) {
							e.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final SynchronizedStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getExpression().accept(this, arg + 1);
					n.getBody().accept(this, arg + 1);
				}

				@Override
				public void visit(final ThisExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getClassExpr().ifPresent(ce -> ce.accept(this, arg + 1));
				}

				@Override
				public void visit(final ThrowStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getExpression().accept(this, arg + 1);
				}

				@Override
				public void visit(final TryStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					// for (final VariableDeclarationExpr v : n.getResources()) { 원래 이 code였는데 밑에 줄로
					// 바꿈.
					for (final Expression v : n.getResources()) {
						v.accept(this, arg + 1);
					}
					n.getTryBlock().accept(this, arg + 1);
					if (n.getCatchClauses() != null) {
						for (final CatchClause c : n.getCatchClauses()) {
							c.accept(this, arg + 1);
						}
					}
					n.getFinallyBlock().ifPresent(f -> f.accept(this, arg + 1));
				}
				/*
				 * @Override public void visit(final TypeDeclarationStmt n, final Integer arg) {
				 *  visitComment(n.getComment(), arg + 1);
				 * n.getTypeDeclaration().accept(this, arg + 1); }
				 */

				@Override
				public void visit(final TypeParameter n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					if (n.getTypeBound() != null) {
						for (final ClassOrInterfaceType c : n.getTypeBound()) {
							c.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final UnaryExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getExpression().accept(this, arg + 1);
				}

				@Override
				public void visit(final UnknownType n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
				}

				@Override
				public void visit(final VariableDeclarationExpr n, final Integer arg) {
					log_flag = false;
					out(n, arg);
					write_logged_method(n.getVariables().toString(),n, arg);
					write_logged_method(n.getCommonType().toString(),n, arg);
					if (!log_flag) {
						visitComment(n.getComment(), arg + 1);
						visitAnnotations(n, arg + 1);
						n.getElementType().accept(this, arg + 1);
						for (final VariableDeclarator v : n.getVariables()) {
							v.accept(this, arg + 1);
						}
					}
				}

				@Override
				public void visit(final VariableDeclarator n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					// n.getId().accept(this, arg + 1);
					n.getInitializer().ifPresent(i -> i.accept(this, arg + 1));
				}
				/*
				 * @Override public void visit(final VariableDeclaratorId n, final Integer arg)
				 * {  visitComment(n.getComment(), arg + 1); }
				 */

				@Override
				public void visit(final VoidType n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);

				}

				@Override
				public void visit(final WhileStmt n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getCondition().accept(this, arg + 1);
					n.getBody().accept(this, arg + 1);
				}

				@Override
				public void visit(final WildcardType n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					visitAnnotations(n, arg + 1);
					n.getExtendedTypes().ifPresent(e -> e.accept(this, arg + 1));
					n.getSuperTypes().ifPresent(s -> s.accept(this, arg + 1));
				}

				@Override
				public void visit(LambdaExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					if (n.getParameters() != null) {
						for (final Parameter a : n.getParameters()) {
							a.accept(this, arg + 1);
						}
					}
					if (n.getBody() != null) {
						n.getBody().accept(this, arg + 1);
					}
				}

				@Override
				public void visit(MethodReferenceExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					n.getTypeArguments().ifPresent(tas -> tas.forEach(ta -> ta.accept(this, arg + 1)));
					if (n.getScope() != null) {
						n.getScope().accept(this, arg + 1);
					}
				}

				@Override
				public void visit(TypeExpr n, final Integer arg) {
					out(n, arg);
					visitComment(n.getComment(), arg + 1);
					if (n.getType() != null) {
						n.getType().accept(this, arg + 1);
					}
				}
				/*
				 * @Override public void visit(ArrayBracketPair n, Integer arg) { 
				 * visitAnnotations(n, arg + 1); }
				 */

				@Override
				public void visit(NodeList n, Integer arg) {
					// 
					for (Object node : n) {
						((Node) node).accept(this, arg + 1);
					}
				}
				/*
				 * @Override public void visit(EmptyImportDeclaration n, Integer arg) { out(n,
				 * arg); visitComment(n.getComment(), arg + 1);
				 * 
				 * }
				 * 
				 * @Override public void visit(SingleStaticImportDeclaration n, Integer arg) {
				 *  visitComment(n.getComment(), arg + 1); n.getType().accept(this,
				 * arg + 1); }
				 * 
				 * @Override public void visit(SingleTypeImportDeclaration n, Integer arg) {
				 *  visitComment(n.getComment(), arg + 1); n.getType().accept(this,
				 * arg + 1); }
				 * 
				 * @Override public void visit(StaticImportOnDemandDeclaration n, Integer arg) {
				 *  visitComment(n.getComment(), arg + 1); n.getType().accept(this,
				 * arg + 1); }
				 * 
				 * @Override public void visit(TypeImportOnDemandDeclaration n, Integer arg) {
				 *  visitComment(n.getComment(), arg + 1); n.getName().accept(this,
				 * arg + 1); }
				 */

				private void visitComment(final Optional<? extends Comment> n, final Integer arg) {
					n.ifPresent(c -> out(c, arg));
					n.ifPresent(c -> c.accept(this, arg + 1));
				}

				private void visitAnnotations(NodeWithAnnotations<?> n, Integer arg) {
					out((Node) n, arg);
					for (AnnotationExpr annotation : n.getAnnotations()) {
						annotation.accept(this, arg + 1);
					}
				}
				public void write_logged_method(String str, Node n, Integer arg) {
					for (int i = 0; i < isLog.length; i++) {
						if (str.toLowerCase().contains(isLog[i])) {
							try {
								// fileWrite.write(indent("\t", arg) + (arg) + "Logged_method ");
								fileWrite.write("-Logged_method");
								fileWrite.flush();
							} catch (IOException e) {
								e.printStackTrace();
							}
							log_flag = true;
							break;
						}
					}

				}

				@Override
				public void visit(ImportDeclaration n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(LocalClassDeclarationStmt n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(Name n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(SimpleName n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(ModuleDeclaration n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(ModuleRequiresDirective n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(ModuleExportsDirective n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(ModuleProvidesDirective n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(ModuleUsesDirective n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(ModuleOpensDirective n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(UnparsableStmt n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(ReceiverParameter n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(VarType n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(Modifier n, Integer arg) {
					// TODO Auto-generated method stub

				}

				@Override
				public void visit(SwitchExpr switchExpr, Integer arg) {
					// TODO Auto-generated method stub

				}
			}, 0);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}
	protected static void printNode(Node n, int indentLevel) {
		Class nodeType = n.getClass();

		String filePath = "G:\\내 드라이브\\Taeha\\Research\\JavaAutoLogging\\JavaParser\\suin\\output\\" + output;
		File file1 = new File(filePath);
		try {
			BufferedWriter fileWrite = new BufferedWriter(new FileWriter(file1, true));
			fileWrite.newLine();
			fileWrite.write(indent("\t", indentLevel) + indentLevel + nodeType.getSimpleName() + " ");
			
			fileWrite.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * @effects generate and return an indent consisting of <tt>level</tt> separator
	 *          chars
	 * @author ducmle
	 */
	private static String indent(String indentChar, int level) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < level; i++) {
			sb.append(indentChar);
		}
		return sb.toString();
	}

	static String repository = "libgdx";
	static String output = repository + "-code.txt";
	static String input = repository;

	public static void main(String[] args) {
		File projectDir = new File("G:\\내 드라이브\\Taeha\\Research\\JavaAutoLogging\\JavaParser\\suin\\input\\" + input);
		try {
			listStruct(projectDir);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
