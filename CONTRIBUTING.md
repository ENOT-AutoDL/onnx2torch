# Contributing

## Introduction

We appreciate your interest in considering contributing to onnx2torch2. Community contributions mean a lot to us.

## Contributions we need

You may already know how you'd like to contribute, whether it's a fix for a bug you
encountered, or a new feature your team wants to use.

If you don't know where to start, consider improving
documentation, bug triaging, and writing tutorials are all examples of
helpful contributions that mean less work for you.

## Your First Contribution

Unsure where to begin contributing? You can start by looking through either of these two issue lists:

* [help-wanted
issues](https://github.com/untetherai/onnx2torch2/issues?q=is%3Aopen+is%3Aissue+label%3ahelp-wanted)
* [good-first-issue issues](https://github.com/untetherai/onnx2torch2/pulls?q=is%3Apr+is%3Aopen+sort%3Aupdated-desc+label%3A%22good+first+issue%22)

Never contributed to open source before? Here are a couple of friendly
tutorials:

-   <http://makeapullrequest.com/>
-   <http://www.firsttimersonly.com/>

## Getting Started

Here's how to get started with your code contribution:

1.  Create your own fork of onnx2torch2
2.  Do the changes in your fork
3.
    *Create a virtualenv and install the development dependencies from the dev_requirements.txt file:*

        a.  python -m venv .venv
        b.  source .venv/bin/activate
        c.  pip install .[dev]

5.  While developing, make sure the tests pass by running `pytest -s`
6.  If you like the change and think the project could use it, send a
    pull request

### Security Vulnerabilities

**NOTE**: If you find a security vulnerability, do NOT open an issue.
Email [Untether Open Source (<oss@untether.ai>)](mailto:oss@untether.ai) instead.

In order to determine whether you are dealing with a security issue, ask
yourself these two questions:

-   Can I access something that's not mine, or something I shouldn't
    have access to?
-   Can I disable something for other people?

If the answer to either of those two questions are *yes*, then you're
probably dealing with a security issue. Note that even if you answer
*no*  to both questions, you may still be dealing with a security
issue, so if you're unsure, just email [us](mailto:oss@untether.ai).

## Suggest a feature or enhancement

If you'd like to contribute a new feature, make sure you check our
issue list to see if someone has already proposed it. Work may already
be underway on the feature you want or we may have rejected a
feature like it already.

If you don't see anything, open a new issue that describes the feature
you would like and how it should work.

## Code review process

The core team regularly looks at pull requests. We will provide
feedback as soon as possible. After receiving our feedback, please respond
within two weeks. After that time, we may close your PR if it isn't
showing any activity.