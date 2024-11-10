class SongNode:
    def __init__(self, title, artist, duration):
        self.title = title
        self.artist = artist
        self.duration = duration
        self.next = None
        self.prev = None

class Playlist:
    def __init__(self):
        self.head = None
        self.tail = None

    def add_song(self, title, artist, duration):
        new_song = SongNode(title, artist, duration)
        if not self.head:
            self.head = self.tail = new_song
        else:
            self.tail.next = new_song
            new_song.prev = self.tail
            self.tail = new_song

    def delete_song(self, title):
        if not self.head:
            print("재생 목록이 비어 있습니다.")
            return

        current = self.head
        while current and current.title != title:
            current = current.next

        if not current:
            print(f"{title}을(를) 찾을 수 없습니다.")
            return

        # 첫 번째 노드 삭제
        if current == self.head:
            self.head = current.next
            if self.head:
                self.head.prev = None
        # 마지막 노드 삭제
        elif current == self.tail:
            self.tail = current.prev
            if self.tail:
                self.tail.next = None
        # 중간 노드 삭제
        else:
            current.prev.next = current.next
            current.next.prev = current.prev

        print(f"{title}이(가) 재생 목록에서 삭제되었습니다.")

    def play_forward(self):
        current = self.head
        while current:
            print(f"재생 중: {current.title} - {current.artist} ({current.duration}분)")
            current = current.next

    def play_backward(self):
        current = self.tail
        while current:
            print(f"재생 중: {current.title} - {current.artist} ({current.duration}분)")
            current = current.prev

# 예제 사용
playlist = Playlist()
playlist.add_song("Song A", "Artist A", 3)
playlist.add_song("Song B", "Artist B", 4)
playlist.add_song("Song C", "Artist C", 5)

print("정방향 재생:")
playlist.play_forward()

print("\n역방향 재생:")
playlist.play_backward()

print("\n곡 삭제:")
playlist.delete_song("Song B")
playlist.play_forward()
