# Collaborative Development Using GitHub and Git

This document provides a step-by-step guide for using GitHub and Git in our project workflow. It covers everything from creating issues to merging pull requests, ensuring all team members can contribute efficiently and effectively.

## Step 1: Create an Issue on GitHub

1. **Navigate to the GitHub repository** where the project is hosted: https://github.com/gromdimon/brain-segment
2. Click on the **Issues** tab next to the repository's name.
3. Press the **New issue** button to start creating a new issue.
4. **Fill in the issue template** with a title that briefly describes the task or bug and a detailed comment section where you can elaborate on what needs to be done or fixed.
5. **Assign the issue** to yourself or another team member, and add any relevant labels (e.g., bug, feature, enhancement).
6. Submit the issue by clicking on **Submit new issue**.

## Step 2: Create a Branch for This Issue

1. In the GitHub UI, navigate to the **Issues** tab and find the issue you just created.
2. Click on the issue to open it and review the details.
3. **Create a new branch** for this issue by clicking on the **Create a branch** link under the `Development` section.
4. Name the branch according to the issue, e.g., `feature/new-feature` or `bugfix/issue-123`.

## Step 3: Checkout Locally to This Branch

1. Open your terminal or Git Bash.
2. Navigate into your repository directory:
   ```
   cd brain-segment
   ```
3. Fetch and checkout your branch. Change `[branch-name]` to the name of the branch you created in Step 2:
   ```
   git fetch origin
   git checkout -b [branch-name] origin/[branch-name]
   ```

## Step 4: Work on This Branch

1. Make the necessary changes to the codebase or documents to address the issue.
2. Test your changes thoroughly to ensure they meet the project's standards and don't introduce new bugs.

## Step 5: Add, Commit, and Push Your Changes

1. Stage your changes for commit:
   ```
   git add .
   ```
   Or add specific files:
   ```
   git add <file1> <file2>
   ```
2. Commit your changes with a meaningful message:
   ```
   git commit -m "A descriptive message explaining what you have done"
   ```
3. Push your changes to the remote repository. Replace `[branch-name]` with the name of your branch:
   ```
   git push origin [branch-name]
   ```

## Step 6: Open a Pull Request (PR)

1. Go to the repository on GitHub.
2. You'll likely see a prompt to **Compare & pull request** for your recently pushed branch. Click on it. If not, navigate to the **Pull requests** tab and click **New pull request**.
3. Select your branch and ensure it's being compared to the correct base branch `main`.
4. Fill in the PR form with a title, description, and any other necessary details about what your code changes entail.
5. Click **Create pull request**.

## Step 7: Review and Merge

1. Notify the project lead or another team member to review your pull request.
2. Address any feedback received. This might involve making further commits to your branch and pushing them.
3. Once approved, the project lead (or someone with merge permissions) will merge your pull request into the base branch.
4. Congratulations! Your contributions are now part of the project. You can safely delete your branch if GitHub offers the option post-merge.

## Additional Notes

- Always ensure your branch is up to date with the base branch before starting work. Use `git pull origin main` (replace `main` with your project's default branch if different) to update your local main branch.
- Use meaningful commit messages that clearly explain the changes you've made. This practice helps in understanding the project's history.

This workflow is designed to promote collaboration and efficiency while minimizing conflicts and confusion. It ensures that all code changes are reviewed and approved, maintaining the quality and integrity of the project.
