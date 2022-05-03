from selenium import webdriver

driver = webdriver.Chrome  # Optional argument, if not specified will search path.
driver.get('http://www.google.com/');
html = driver.page_source
f = open("myhtml", "wt")
f.write(html)
f.close()