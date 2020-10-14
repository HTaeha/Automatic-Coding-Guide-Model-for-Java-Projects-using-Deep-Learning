

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

public class JAE {
	public static void listStruct(File projectDir) throws FileNotFoundException {
		new DirExplorer((level, path, file) -> path.endsWith("-partial_code.java"), (level, path, file) -> {
			//print(file, path);
		}).explore(projectDir);
	}

	private static final Class[] CompositeNodeClasses = {
			// CompilationUnit.class,
			// ClassOrInterfaceDeclaration.class,
			MethodDeclaration.class, FieldDeclaration.class, ExpressionStmt.class, VariableDeclarationExpr.class,
			BlockStmt.class, IfStmt.class, WhileStmt.class, ForEachStmt.class, ForStmt.class, DoStmt.class,
			SwitchStmt.class,
			// SwitchEntryStmt.class,
			TryStmt.class, CatchClause.class, ThrowStmt.class, ReturnStmt.class, };
	private static final Class[] FeaturedNodeClasses = { IfStmt.class, MethodDeclaration.class, FieldDeclaration.class,
			ExpressionStmt.class, VariableDeclarationExpr.class, WhileStmt.class, ForEachStmt.class, ForStmt.class,
			DoStmt.class, SwitchStmt.class, TryStmt.class, CatchClause.class, ThrowStmt.class, ReturnStmt.class, };

	/**
	 * @effects Parse <tt>code</tt> and print its abstract syntax tree (AST) out on
	 *          the standard output.
	 * 
	 * @author adapted from code at
	 *         {@link https://github.com/javaparser/javaparser/issues/538#issuecomment-276155353}
	 *         by Danny van Bruggen (matozoid)
	 */
	public static void print(File file, String path, BufferedWriter fileWrite, BufferedWriter fileWrite_AST, String method_name) throws IOException {
		CompilationUnit cu = null;
		try {
			cu = JavaParser.parse(file);
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		
		String[] isException = new String[2];
		isException[0] = "throw";
		isException[1] = "exception";

		cu.accept(new VoidVisitor<Integer>() {
			boolean exception_flag = false;

			public void out(Node n, int indentLevel) {
				printNode(n, indentLevel, fileWrite_AST);
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

				if (print_type == 2) {
					String catchParam = n.getParameter().toString().replaceAll("\n|\r|\\[|\\]|\"", " ");
					try {
						fileWrite_AST.write(catchParam);
						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

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

				if (print_type == 2) {
					try {
						fileWrite_AST.write(n.getCondition().toString().replaceAll("\n|\r|\\[|\\]|\"", " "));

						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

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
			 * arg) { out(n, arg); visitComment(n.getComment(), arg + 1); }
			 */

			@Override
			public void visit(final EmptyStmt n, final Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);
			}
			/*
			 * @Override public void visit(final EmptyTypeDeclaration n, final Integer arg)
			 * { out(n, arg); visitComment(n.getComment(), arg + 1);
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
				out(n, arg);

				if (print_type == 2) {
					try {
						exception_flag = false;
						for (int i = 0; i < isException.length; i++) {
							// throw method이면 expression을 write하지 않음.
							if (n.toString().toLowerCase().contains(isException[i])) {
								exception_flag = true;
								break;
							}
						}
						if (!exception_flag) {
							String part = n.toString().replaceAll("\n|\r|\\[|\\]|\"", " ").split("//")[0];
							String[] parts = part.split("\"");
							for (int j = 0; j < parts.length; j++) {
								fileWrite_AST.write(parts[j] + ' ');
								fileWrite_AST.flush();
							}

							fileWrite_AST.flush();
						}
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

				//if(!n.toString().toLowerCase().contains("throw")) {
					visitComment(n.getComment(), arg + 1);
					n.getExpression().accept(this, arg + 1);
				//}					
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
				out(n, arg);

				exception_flag = false;
				for (int i = 0; i < isException.length; i++) {
					if (n.getVariables().toString().toLowerCase().replaceAll("\n|\r|\\[|\\]|\"", " ").contains(isException[i]) ||
							n.getModifiers().toString().toLowerCase().replaceAll("\n|\r|\\[|\\]|\"", " ").contains(isException[i])) {
						try {
							fileWrite_AST.newLine();
							fileWrite_AST.write(indent("\t", arg + 1) + (arg + 1) + "EXCEPTION");
							fileWrite_AST.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}
						exception_flag = true;
					}
				}
				if (print_type == 2 && !exception_flag) {
					try {
						fileWrite_AST.write(
								n.getModifiers().toString().replaceAll("\n|\r|\\[|\\]|\"", " ").replace(",", "")
										+ " " + n.getVariables().toString().replaceAll("\n|\r|\\[|\\]|\"", " "));

						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

				/*
				 * visitComment(n.getComment(), arg + 1); visitAnnotations(n, arg + 1);
				 * n.getElementType().accept(this, arg + 1); for (final VariableDeclarator var :
				 * n.getVariables()) { var.accept(this, arg + 1); }
				 */
			}

			@Override
			public void visit(final ForEachStmt n, final Integer arg) {
				out(n, arg);

				if (print_type == 2) {
					try {
						fileWrite_AST.write(n.getVariable().toString().replaceAll("\n|\r|\\[|\\]|\"", " ") + " : "
								+ n.getIterable().toString().replaceAll("\n|\r|\\[|\\]|\"", " "));

						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

				visitComment(n.getComment(), arg + 1);
				n.getVariable().accept(this, arg + 1);
				n.getIterable().accept(this, arg + 1);
				n.getBody().accept(this, arg + 1);
			}

			@Override
			public void visit(final ForStmt n, final Integer arg) {
				out(n, arg);

				if (print_type == 2) {
					String init = n.getInitialization().toString().replaceAll("\n|\r|\\[|\\]|\"", " ");
					String comp = n.getCompare().toString().replaceAll("\n|\r|\\[|\\]|\"", " ").replace("Optional",
							"");
					String update = n.getUpdate().toString().replaceAll("\n|\r|\\[|\\]|\"", " ");

					try {
						fileWrite_AST.write(init + " ; " + comp + " ; " + update);

						fileWrite_AST.flush();

					} catch (IOException e) {
						e.printStackTrace();
					}
				}

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

				if (print_type == 2) {
					String ifCondition = n.getCondition().toString();
					String[] parts = ifCondition.split("\"");

					for (int i = 0; i < parts.length; i++) {
						String[] part = parts[i].split(" ");
						for (int j = 0; j < part.length; j++) {
							try {
								fileWrite_AST.write(part[j] + " ");
								fileWrite_AST.flush();
							} catch (IOException e) {
								e.printStackTrace();
							}
						}

					}
				}
				try {

					fileWrite_AST.flush();
				} catch (IOException e) {
					e.printStackTrace();
				}

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
			 * arg) { out(n, arg); visitComment(n.getComment(), arg + 1); }
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
			 * arg) { out(n, arg); visitComment(n.getComment(), arg + 1); }
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
				out(n, arg);
				for (int i = 0; i < isException.length; i++) {
					if (n.getNameAsExpression().toString().toLowerCase().contains(isException[i])) {
						try {
							fileWrite_AST.newLine();
							fileWrite_AST.write(indent("\t", arg + 1) + (arg + 1) + "EXCEPTION");
							fileWrite_AST.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
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

			@Override
			public void visit(final MethodDeclaration n, final Integer arg) {
				if (n.getNameAsString().equals(method_name)) {
					try {
						fileWrite.write(n.getDeclarationAsString());
						fileWrite.newLine();
						fileWrite.flush();
						fileWrite.write(String.valueOf(n.getBody()));
						fileWrite.newLine();
						fileWrite.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
				try {
					fileWrite_AST.newLine();
					fileWrite_AST.write("path: " + path);
					fileWrite_AST.newLine();
					fileWrite_AST.flush();
					fileWrite_AST.write("method: " + n.getNameAsString());
					fileWrite_AST.flush();
				} catch (IOException e1) {
					e1.printStackTrace();
				}
				out(n, arg);
				
				if (print_type == 2) {
					try {
						fileWrite_AST.write(n.getDeclarationAsString().split(" ")[0] + " ");
						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
					String[] parts = n.getTypeAsString().toString().replaceAll("\n|\r|\\[|\\]|\"", " ").split(" ");
					exception_flag = false;
					for (int j = 0; j < parts.length; j++) {
						
						for (int i = 0; i < isException.length; i++) {
							if (parts[j].toLowerCase().contains(isException[i])) {
								exception_flag = true;
								break;
							}
							if (i == isException.length - 1 && !exception_flag) {
								try {
									fileWrite_AST.write(parts[j] + " ");
									fileWrite_AST.flush();
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
						}
						exception_flag = false;
					}
					try {
						fileWrite_AST.write(n.getNameAsString() + " ");
						fileWrite_AST.flush();
					} catch (IOException e1) {
						e1.printStackTrace();
					}
					String[] parts1 = n.getParameters().toString().replaceAll("\n|\r|\\[|\\]|\"", " ").split(" ");

					for (int j = 0; j < parts1.length; j++) {
						for (int i = 0; i < isException.length; i++) {
							if (parts1[j].toLowerCase().contains(isException[i])) {
								exception_flag = true;
								break;
							}
							if (i == isException.length - 1 && !exception_flag) {
								try {
									fileWrite_AST.write(parts1[j] + " ");
									fileWrite_AST.flush();
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
						}
						exception_flag = false;
					}
					/*try {
						fileWrite_AST.write(n.getDeclarationAsString().replaceAll("\n|\r|\\[|\\]|\"", " "));
						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}*/
				}
				String method_check = n.getDeclarationAsString().replaceAll("\n|\r|\\[|\\]|\"", " ");
				if(method_check.contains("throws")) {
					try {
						fileWrite_AST.newLine();
						fileWrite_AST.write(indent("\t", arg + 1) + (arg + 1) + "EXCEPTION");
						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
				/*
				 * visitComment(n.getComment(), arg + 1); visitAnnotations(n, arg + 1); if
				 * (n.getTypeParameters() != null) { for (final TypeParameter t :
				 * n.getTypeParameters()) { t.accept(this, arg + 1); } }
				 * n.getType().accept(this, arg + 1); n.getName().accept(this, arg + 1); if
				 * (n.getParameters() != null) { for (final Parameter p : n.getParameters()) {
				 * p.accept(this, arg + 1); } }
				 */ /*
					 * if (n.getThrows() != null) { for (final ReferenceType name : n.getThrows()) {
					 * name.accept(this, arg + 1); } }
					 */
				n.getBody().ifPresent(b -> b.accept(this, arg + 1));
			}

			@Override
			public void visit(final NameExpr n, final Integer arg) {
				out(n, arg);
				for (int i = 0; i < isException.length; i++) {
					if (n.getNameAsExpression().toString().toLowerCase().contains(isException[i])) {
						try {
							fileWrite_AST.newLine();
							fileWrite_AST.write(indent("\t", arg + 1) + (arg + 1) + "EXCEPTION");
							fileWrite_AST.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
				visitComment(n.getComment(), arg + 1);
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
			 * out(n, arg); visitComment(n.getComment(), arg + 1);
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
				out(n, arg);

				exception_flag = false;
				for (int i = 0; i < isException.length; i++) {
					if (n.getExpression().toString().toLowerCase().replace("Optional[", "").replaceAll("\n|\r|\\[|\\]|\"", " ")
							.contains(isException[i])) {
						try {
							fileWrite_AST.newLine();
							fileWrite_AST.write(indent("\t", arg + 1) + (arg + 1) + "EXCEPTION");
							fileWrite_AST.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}
						exception_flag = true;
					}
				}
				if (print_type == 2 && !exception_flag) {
					String ret = n.getExpression().toString().replace("Optional[", "")
							.replaceAll("\n|\r|\\[|\\]|\"", " ");
					try {
						fileWrite_AST.write(ret);
						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

				visitComment(n.getComment(), arg + 1);
				n.getExpression().ifPresent(e -> e.accept(this, arg + 1));
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

				if (print_type == 2) {
					try {
						fileWrite_AST.write(n.getSelector().toString());

						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

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

				if (print_type == 2) {
					String throwParam = n.getExpression().toString().replaceAll("\n|\r|\\[|\\]|\"", " ");
					String[] parts = throwParam.split("\\("); // parts[0] : exception 부분까지만 추출.
					try {
						// fileWrite_AST.write(indent + "THROW" + " " + parts[0]);
						fileWrite_AST.write(throwParam);

						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

				visitComment(n.getComment(), arg + 1);
				n.getExpression().accept(this, arg + 1);
			}

			@Override
			public void visit(final TryStmt n, final Integer arg) {
				out(n, arg);

				if (print_type == 2) {
					try {
						fileWrite_AST.write(n.getResources().toString().replaceAll("\n|\r|\\[|\\]|\"", " "));

						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

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
			 * out(n, arg); visitComment(n.getComment(), arg + 1);
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
				out(n, arg);

				exception_flag = false;
				for (int i = 0; i < isException.length; i++) {
					if (n.getVariables().toString().toLowerCase().replaceAll("\n|\r|\\[|\\]|\"", " ").contains(isException[i]) || 
							n.getCommonType().toString().toLowerCase().replaceAll("\n|\r|\\[|\\]|\"", " ").contains(isException[i])) {
						try {
							fileWrite_AST.newLine();
							fileWrite_AST.write(indent("\t", arg + 1) + (arg + 1) + "EXCEPTION");
							fileWrite_AST.flush();
						} catch (IOException e) {
							e.printStackTrace();
						}
						exception_flag = true;
					}
				}
				if (print_type == 2 && !exception_flag) {
					try {
						fileWrite_AST.write(n.getCommonType().toString().replaceAll("\n|\r|\\[|\\]|\"", " ")
								+ n.getVariables().toString().replaceAll("\n|\r|\\[|\\]|\"", " "));

						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

				visitComment(n.getComment(), arg + 1);
				visitAnnotations(n, arg + 1);
				n.getElementType().accept(this, arg + 1);
				for (final VariableDeclarator v : n.getVariables()) {
					v.accept(this, arg + 1);
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
			 * { out(n, arg); visitComment(n.getComment(), arg + 1); }
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

				if (print_type == 2) {
					try {
						fileWrite_AST.write(n.getCondition().toString().replaceAll("\n|\r|\\[|\\]|\"", " "));

						fileWrite_AST.flush();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

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
			 * @Override public void visit(ArrayBracketPair n, Integer arg) { out(n, arg);
			 * visitAnnotations(n, arg + 1); }
			 */

			@Override
			public void visit(NodeList n, Integer arg) {
				// out(n, arg);
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
			 * out(n, arg); visitComment(n.getComment(), arg + 1); n.getType().accept(this,
			 * arg + 1); }
			 * 
			 * @Override public void visit(SingleTypeImportDeclaration n, Integer arg) {
			 * out(n, arg); visitComment(n.getComment(), arg + 1); n.getType().accept(this,
			 * arg + 1); }
			 * 
			 * @Override public void visit(StaticImportOnDemandDeclaration n, Integer arg) {
			 * out(n, arg); visitComment(n.getComment(), arg + 1); n.getType().accept(this,
			 * arg + 1); }
			 * 
			 * @Override public void visit(TypeImportOnDemandDeclaration n, Integer arg) {
			 * out(n, arg); visitComment(n.getComment(), arg + 1); n.getName().accept(this,
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

			@Override
			public void visit(ImportDeclaration n, Integer arg) {
				// TODO Auto-generated method stub
				out(n, arg);
				visitComment(n.getComment(), arg + 1);
			}
			@Override
			public void visit(LocalClassDeclarationStmt n, Integer arg) {
				// TODO Auto-generated method stub
				out(n, arg);
				visitComment(n.getComment(), arg + 1);
			}

			@Override
			public void visit(Name n, Integer arg) {
				// TODO Auto-generated method stub
				out(n, arg);
				visitComment(n.getComment(), arg + 1);
			}

			@Override
			public void visit(SimpleName n, Integer arg) {
				// TODO Auto-generated method stub
				out(n, arg);
				visitComment(n.getComment(), arg + 1);
			}

			@Override
			public void visit(ModuleDeclaration n, Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);

			}

			@Override
			public void visit(ModuleRequiresDirective n, Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);

			}

			@Override
			public void visit(ModuleExportsDirective n, Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);

			}

			@Override
			public void visit(ModuleProvidesDirective n, Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);

			}

			@Override
			public void visit(ModuleUsesDirective n, Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);

			}

			@Override
			public void visit(ModuleOpensDirective n, Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);

			}

			@Override
			public void visit(UnparsableStmt n, Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);

			}

			@Override
			public void visit(ReceiverParameter n, Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);

			}

			@Override
			public void visit(VarType n, Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);

			}

			@Override
			public void visit(Modifier n, Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);

			}

			@Override
			public void visit(SwitchExpr n, Integer arg) {
				out(n, arg);
				visitComment(n.getComment(), arg + 1);

			}
		}, 0);
	}

	/**
	 * @effects Print to the standard output the user-friendly content of <tt>n</tt>
	 * 
	 * @version
	 */
	protected static void printNode(Node n, int indentLevel, BufferedWriter fileWrite_AST) {
		Class nodeType = n.getClass();

		try {

			if (isComposite(n)) {
				// n is a composite node -> print the type name only

				fileWrite_AST.newLine();
				fileWrite_AST.write(indent("\t", indentLevel) + indentLevel + nodeType.getSimpleName() + " ");
				fileWrite_AST.flush();
			} else {
				// no is non-composite -> print the type name + content
				// String content = n.toStringWithoutComments();
				// System.out.println(indent(" ", indentLevel) + nodeType.getSimpleName() + ": "
				// + content);

				if (AST_type == 1) {
					fileWrite_AST.newLine();
					fileWrite_AST.write(indent("\t", indentLevel) + indentLevel + nodeType.getSimpleName() + " ");
					fileWrite_AST.flush();
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * @effects if n is a composite node (i.e. its type is one of those specified in
	 *          {@link #CompositeNodeClasses} return true else return false
	 * @version
	 * 
	 */
	private static boolean isComposite(Node n) {
		Class nodeType = n.getClass();

		for (Class c : CompositeNodeClasses) {
			if (c.equals(nodeType))
				return true;
		}

		// non-composite
		return false;
	}

	private static boolean isFeaturedClass(Node n) {
		Class nodeType = n.getClass();

		for (Class c : FeaturedNodeClasses) {
			if (c.equals(nodeType))
				return true;
		}

		// non-featured
		return false;
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

	static int AST_type = 1; // 1 : AST
								// 2 : simpleAST
	static int print_type = 1; // 1 : no feature, only AST.
								// 2 : feature with AST.

}
