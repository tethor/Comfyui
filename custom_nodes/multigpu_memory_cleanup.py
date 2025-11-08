"""
DisTorch2 Memory Cleanup Node for ComfyUI

This custom node provides memory cleanup functionality specifically for
MultiGPU DisTorch2 setups to prevent memory leaks with large models.

Author: tethor
Date: 2025-11-08
"""

import logging
import torch
import gc
import comfy.model_management


def cleanup_distorch_metadata():
    """
    Clean DisTorch2 metadata from loaded models to prevent memory leaks.

    This function removes DisTorch2-specific attributes that prevent proper
    garbage collection of model memory, particularly for large models like Qwen.

    Returns:
        int: Number of models cleaned
    """
    cleaned_count = 0

    try:
        # Iterate through all currently loaded models
        for lm in comfy.model_management.current_loaded_models:
            try:
                # Check for nested model structure (common in DisTorch2)
                if hasattr(lm.model, 'model'):
                    inner = lm.model.model

                    # Remove DisTorch2 metadata
                    if hasattr(inner, '_distorch_v2_meta'):
                        delattr(inner, '_distorch_v2_meta')
                        cleaned_count += 1

                    # Remove unload markers
                    if hasattr(inner, '_mgpu_unload_distorch_model'):
                        delattr(inner, '_mgpu_unload_distorch_model')

                # Remove block assignments
                if hasattr(lm.model, '_distorch_block_assignments'):
                    delattr(lm.model, '_distorch_block_assignments')

            except Exception as e:
                logging.debug(f"[DisTorch2 Cleanup] Error cleaning model: {e}")
                continue

        if cleaned_count > 0:
            logging.info(f"[DisTorch2 Cleanup] Cleaned metadata from {cleaned_count} models")

    except Exception as e:
        logging.warning(f"[DisTorch2 Cleanup] Error during cleanup: {e}")

    return cleaned_count


def release_multigpu_memory():
    """
    Release pinned memory across all CUDA devices.

    Iterates through all available CUDA devices and performs memory cleanup
    operations including synchronization, cache clearing, and IPC collection.
    """
    if not torch.cuda.is_available():
        return

    try:
        device_count = torch.cuda.device_count()

        for device_id in range(device_count):
            try:
                with torch.cuda.device(device_id):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception as e:
                logging.debug(f"[DisTorch2 Cleanup] Error cleaning device {device_id}: {e}")

        # Perform IPC collection and reset stats
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

        logging.debug(f"[DisTorch2 Cleanup] Cleaned memory across {device_count} CUDA devices")

    except Exception as e:
        logging.warning(f"[DisTorch2 Cleanup] Error during multi-GPU memory release: {e}")


class DisTorch2MemoryCleanup:
    """
    ComfyUI node for cleaning DisTorch2 metadata and releasing multi-GPU memory.

    This node can be inserted in workflows after generation steps to ensure
    proper memory cleanup, particularly important for large models using
    MultiGPU/DisTorch2 distribution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": ("*",),
            },
            "optional": {
                "force_gc": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "cleanup_memory"
    CATEGORY = "utils/memory"
    OUTPUT_NODE = False

    def cleanup_memory(self, any_input, force_gc=True):
        """
        Execute memory cleanup operations.

        Args:
            any_input: Any input type (passed through)
            force_gc: Whether to force garbage collection (default: True)

        Returns:
            tuple: The input passed through unchanged
        """
        try:
            # Clean DisTorch2 metadata
            cleaned_count = cleanup_distorch_metadata()

            # Release multi-GPU memory
            release_multigpu_memory()

            # Force garbage collection if requested
            if force_gc:
                gc.collect(0)
                gc.collect(1)
                gc.collect(2)

            if cleaned_count > 0:
                logging.info(f"[DisTorch2 Cleanup Node] Successfully cleaned {cleaned_count} models")

        except Exception as e:
            logging.error(f"[DisTorch2 Cleanup Node] Error during cleanup: {e}")

        # Pass through the input unchanged
        return (any_input,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DisTorch2MemoryCleanup": DisTorch2MemoryCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DisTorch2MemoryCleanup": "DisTorch2 Memory Cleanup",
}
