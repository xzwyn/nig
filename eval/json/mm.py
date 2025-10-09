python main.py input/en_se.pdf input/de_se.pdf input/english.json input/german.json --evaluate --debug-report
--- Document Alignment Pipeline Started (ToC-First Approach) ---
Step 1/5: Processing full JSON files...

Step 2/5: Extracting and structuring Tables of Contents from PDFs...

Step 3/5: Aligning ToC sections semantically...
Traceback (most recent call last):
  File "C:\Users\M3ZEDTZ\Downloads\test_1\main.py", line 117, in <module>
    main()
    ~~~~^^
  File "C:\Users\M3ZEDTZ\Downloads\test_1\main.py", line 52, in main
    aligned_sections = align_tocs(english_toc, german_toc)
  File "C:\Users\M3ZEDTZ\Downloads\test_1\src\alignment\toc_aligner.py", line 25, in align_tocs
    english_embeddings_list = get_embeddings(eng_titles)
  File "C:\Users\M3ZEDTZ\Downloads\test_1\src\clients\azure_client.py", line 57, in get_embeddings
    response = client.embeddings.create(
        input=texts,
        model=deployment
    )
  File "C:\Users\M3ZEDTZ\AppData\Roaming\Python\Python313\site-packages\openai\resources\embeddings.py", line 132, in create
    return self._post(
           ~~~~~~~~~~^
        "/embeddings",
        ^^^^^^^^^^^^^^
    ...<8 lines>...
        cast_to=CreateEmbeddingResponse,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\M3ZEDTZ\AppData\Roaming\Python\Python313\site-packages\openai\_base_client.py", line 1259, in post     
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\M3ZEDTZ\AppData\Roaming\Python\Python313\site-packages\openai\_base_client.py", line 1047, in request  
    raise self._make_status_error_from_response(err.response) from None
openai.AuthenticationError: Error code: 401 - {'error': {'code': '401', 'message': 'Access denied due to invalid subscription key or wrong API endpoint. Make sure to provide a valid key for an active subscription and use a correct regional API endpoint for your resource.'}}
