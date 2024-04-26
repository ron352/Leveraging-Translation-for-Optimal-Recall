A central retrieval concern emerges regarding the predominant BM25 ranking algorithm’s reliance on exact term
matching, potentially overlooking relevant documents expressing similar concepts using non-identical vocabulary.
While query expansion and document re-ranking techniques can help retrieve additional relevant items, they often
fail to adequately improve recall. Our language model personalization frameworks may face analogous challenges,
as dependency solely on term matching risks missing personalized documents with variant phrasing still useful for
understanding user interests. We propose integrating data augmentation approaches to query expansion to help address
these term mismatch problems through enhanced representation of the searcher’s context.
