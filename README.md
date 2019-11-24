# MLNLP_GroupProj
필요 라이브러리: numpy, pandas, statsmodels

- https://fasttext.cc/docs/en/crawl-vectors.html 에서 korean text (bin 말고)를 다운받고 압축을 풀어서 data 폴더에 넣어주세요.
- data 폴더의 women_job_statistics 에는 첫째 행에 직업군 이름(전문직, 서비스직 등)을 써주시고, 두번째 행에 참여율 수치를 넣어주세요.
- job_words는 첫번째 행에 직업군 이름 (위 women_job_statistics 파일에 썼던 직업군 이름과 동일해야 함)을 적어주시고, 나머지 행에 word vector로 사용할 단어들을 쭉 적어주시면 됩니다.
- regress.py를 실행시키면 regression의 결과가 나오는데, https://datatofish.com/statsmodels-linear-regression/ 여길 참고해서 해석해주세요. p>|t|와 R-squared를 보면 될 것 같습니다.
