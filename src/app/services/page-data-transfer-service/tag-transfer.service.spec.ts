import { TestBed } from '@angular/core/testing';

import { TagTransferService } from './tag-transfer.service';

describe('TagTransferService', () => {
  let service: TagTransferService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(TagTransferService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
