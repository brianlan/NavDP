import argparse

import onnx
from onnx import helper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Patch NavDP denoiser ONNX MultiheadAttention If nodes into static Squeeze nodes for TensorRT."
    )
    parser.add_argument("--input", required=True, help="Input denoiser ONNX path.")
    parser.add_argument("--output", required=True, help="Patched ONNX path.")
    return parser.parse_args()


def is_mha_squeeze_if(node):
    if node.op_type != "If" or "multihead_attn" not in node.name:
        return False
    branch_ops = {
        attr.name: [branch_node.op_type for branch_node in attr.g.node]
        for attr in node.attribute
        if attr.type == onnx.AttributeProto.GRAPH
    }
    return branch_ops.get("then_branch") == ["Constant", "Squeeze"] and branch_ops.get("else_branch") == ["Identity"]


def find_then_squeeze(node):
    constant = None
    squeeze = None
    for attr in node.attribute:
        if attr.name != "then_branch":
            continue
        for branch_node in attr.g.node:
            if branch_node.op_type == "Constant":
                constant = branch_node
            if branch_node.op_type == "Squeeze":
                squeeze = branch_node
    if constant is None or squeeze is None:
        raise ValueError(f"No then-branch Constant/Squeeze found for {node.name}")
    return constant, squeeze


def main():
    args = parse_args()
    model = onnx.load(args.input)

    patched_nodes = []
    replacement_count = 0
    for node in model.graph.node:
        if not is_mha_squeeze_if(node):
            patched_nodes.append(node)
            continue

        constant, squeeze = find_then_squeeze(node)
        axes_output = f"{node.output[0]}_static_squeeze_axes"
        patched_nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[axes_output],
                name=f"{node.name}_static_squeeze_axes",
                value=helper.get_attribute_value(constant.attribute[0]),
            )
        )
        patched_nodes.append(
            helper.make_node(
                "Squeeze",
                inputs=[squeeze.input[0], axes_output],
                outputs=list(node.output),
                name=f"{node.name}_static_squeeze",
            )
        )
        replacement_count += 1

    del model.graph.node[:]
    model.graph.node.extend(patched_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"patched {replacement_count} MultiheadAttention If nodes")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
