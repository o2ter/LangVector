export class BuiltinRule {

  content: string;
  deps: string[];

  constructor(content: string, deps: string[] = []) {
    this.content = content;
    this.deps = deps;
  }
}
