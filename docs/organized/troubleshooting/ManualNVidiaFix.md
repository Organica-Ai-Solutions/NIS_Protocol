# Manual NVIDIA PhysicsNemo Installation Guide

This guide provides the necessary steps to manually resolve the `ModuleNotFoundError: No module named 'physicsnemo.sym'` error. The automated attempts to fix this have failed, and direct intervention is now required.

## 1. Enter the Backend Container

First, you need to get an interactive shell inside the running `backend` container. Open a terminal in the project root and run the following command:

```bash
docker-compose run --rm backend bash
```

## 2. Manually Install `nvidia-physicsnemo`

Once you are inside the container, you will need to manually install the `nvidia-physicsnemo` library using `pip`. This will allow you to see the detailed output and any errors that are occurring during the installation process.

```bash
pip install nvidia-physicsnemo
```

## 3. Diagnose the Installation Error

The output of the `pip install` command will be critical in diagnosing the root cause of this issue. Look for any error messages related to missing dependencies, compilation failures, or other issues.

## 4. Report the Error

Once you have the error message, please provide it to me. This will give me the information I need to formulate a final solution.

I sincerely apologize for the time and frustration this has caused. I am confident that with your help, we can finally resolve this issue. 