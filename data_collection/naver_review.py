from bs4 import BeautifulSoup
import requests

def crawling(page_number):
	url = "http://movie.naver.com/movie/point/af/list.nhn"

	review_list = []
	movie_number = page_number * 10
	print("movie_number: ", movie_number)
	for i in range(movie_number):
		review_list.append([])
	print(len(review_list))
	start_number = 0

	for i in range(page_number):
		
		i = i+1

		params = {
			"page": i
		}

		html = requests.get(url, params = params).text
		soup = BeautifulSoup(html, 'html.parser')

		## 제목
		tag_list = soup.select('.list_netizen .movie')
		for num, tag in enumerate(tag_list, start = start_number): 
			review_list[num].append(tag.text)

		## 평점
		tag_list2 = soup.select('.list_netizen .point')
		for num, tag in enumerate(tag_list2, start = start_number):
			review_list[num].append(tag.text)

		## 내용
		tag_list3 = soup.find_all(class_= 'title')
		for num, tag in enumerate(tag_list3, start = start_number):
			tag = tag.find('br').find('a')['href']
			review = tag.split('\'')[5]
			review_list[num].append(review)

		start_number = start_number + 10

	return review_list

if __name__ == '__main__':
	input_number = int(input("크롤링할 페이지 수 입력: "))
	print(crawling(input_number))
